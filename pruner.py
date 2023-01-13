import copy 
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np

def pruning_model(model, px):

    print('start unstructured pruning')
    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def pruning_model_random(model, px):

    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    )

def prune_model_custom(model, mask_dict):

    print('start unstructured pruning with custom mask')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'].to(m.weight.data.device))

def remove_prune(model):
    
    print('remove pruning')
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.remove(m,'weight')

def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict

def check_sparsity(model):
    
    sum_list = 0
    zero_sum = 0

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sum_list = sum_list+float(m.weight.nelement())
            zero_sum = zero_sum+float(torch.sum(m.weight == 0))  

    print('* remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
    return 100*(1-zero_sum/sum_list)

def check_sparsity_mask(mask_dict):

    sum_list = 0
    zero_sum = 0

    for key in mask_dict.keys():
        sum_list = sum_list+float(mask_dict[key].nelement())
        zero_sum = zero_sum+float(torch.sum(mask_dict[key] == 0))  

    print('* remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
    return 100*(1-zero_sum/sum_list)

def pruning_with_rewind(model, px, init, mode='l1'):

    if mode == 'l1':
        print('using L1 magnitude pruning')
        pruning_model(model, px)
    elif mode == 'random':
        print('using random pruning')
        pruning_model_random(model, px)
    else:
        raise ValueError('Unsupport pruning mode')

    current_mask = extract_mask(model.state_dict())
    remove_prune(model)

    model.load_state_dict(init)
    prune_model_custom(model, current_mask)
    check_sparsity(model)

def need_to_prune(name,m,c):
    return isinstance(m, nn.Conv2d)

def custom_prune(model, mask_dict, prune_type, num_paths, args, add_back=False):
    new_mask_dict = globals()['prune_' + prune_type](model, mask_dict, num_paths, args)
    n_zeros = 0
    n_param = 0
    n_after_zeros = 0
    for name,m in model.named_modules():
        if need_to_prune(name, m, None):
            mask = mask_dict[name+'.weight_mask']
            n_zeros += (mask == 0).float().sum().item()
            n_param += mask.numel()
            n_after_zeros += (new_mask_dict[name+'.weight_mask'] == 0).float().sum().item()
    print("Sparsity before: {}%".format((1 - n_zeros / n_param) * 100))
    print("Sparsity after: {}%".format((1 - n_after_zeros / n_param) * 100))
    
    if add_back:
        mask_vector = torch.zeros(n_param)
        n_cur = 0
        for name,m in model.named_modules():
            if need_to_prune(name, m, None):
                mask = new_mask_dict[name+'.weight_mask']
                size = np.product(np.array(mask.shape))
                mask_vector[n_cur:n_cur+size] = mask.view(-1)
                n_cur += size
        rand_vector = torch.randn(n_param)
        rand_vector[mask_vector == 1] = np.inf
        threshold, _ = torch.kthvalue(rand_vector, int(n_after_zeros - n_zeros))
        mask_vector[rand_vector < threshold] = 1
        n_cur = 0
        for name,m in model.named_modules():
            if need_to_prune(name, m, None):
                mask = new_mask_dict[name+'.weight_mask']
                size = np.product(np.array(mask.shape))
                new_mask = mask_vector[n_cur:n_cur+size].view(mask.shape).to(mask.device)
                n_cur += size
                m.weight.data = torch.where((new_mask - mask).bool(), torch.randn(mask.shape, device=mask.device) / 100, m.weight.data)
                prune.CustomFromMask.apply(m, 'weight', mask=new_mask)
    else:
        for name,m in model.named_modules():
            if need_to_prune(name, m, None):
                mask = new_mask_dict[name + '.weight_mask']
                prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))

def prune_random_path(model, mask_dict, num_paths, args):
    new_mask_dict = copy.deepcopy(mask_dict)
    for _ in range(num_paths):
        end_index = None
        for name,m in model.named_modules():
            if need_to_prune(name, m, None):
                mask = mask_dict[name+'.weight_mask']
                weight = m.weight * mask
                weight = torch.sum(weight.abs(), [2,3]).cpu().detach().numpy()
                if end_index is None:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                try:
                    prob = np.abs(weight[:, start_index]) > 0
                except IndexError:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = np.abs(weight[:, start_index]) > 0
                prob = prob / (prob.sum() + 1e-10)

                counter = 0
                while prob.sum() == 0:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = np.abs(weight[:, start_index]) > 0
                    prob = prob / (prob.sum() + 1e-10)
                    counter = counter + 1
                    
                    if counter > 200000:
                        prob = np.ones(prob.shape)
                        prob = prob / prob.sum()

                end_index = np.random.choice(np.arange(weight.shape[0]), 1,
                            p=np.array(prob))[0]
                new_mask_dict[name+'.weight_mask'][end_index, start_index, :, :] = 0
                start_index = end_index
    return new_mask_dict

def prune_ewp(model, mask_dict, num_paths, args):
    new_mask_dict = copy.deepcopy(mask_dict)       
    for _ in range(num_paths):
        end_index = None
        for name,m in model.named_modules():
            if need_to_prune(name, m, None):
                weight = m.weight * mask_dict[name+'.weight_mask'] 
                weight = torch.sum(weight.abs(), [2,3]).cpu().detach().numpy()
                if end_index is None:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                try:
                    prob = np.abs(weight[:, start_index])
                except:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = np.abs(weight[:, start_index])
                prob = prob / (prob.sum() + 1e-10)
                
                counter = 0
                while prob.sum() == 0:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = np.abs(weight[:, start_index])
                    prob = prob / (prob.sum() + 1e-10)
                    counter = counter + 1

                    if counter > 200000:
                        prob = np.ones(prob.shape) / np.sum(np.ones(prob.shape))
                
                end_index = np.random.choice(np.arange(weight.shape[0]), 1,
                            p=np.array(prob))[0]
                new_mask_dict[name+'.weight_mask'][end_index, start_index, :, :] = 0
                start_index = end_index

    return new_mask_dict
 
def prune_betweenness(model, mask_dict, num_paths, args, downsample=100):
    new_mask_dict = copy.deepcopy(mask_dict)
    graph = networkx.Graph()
    name_list = []

    for name,m in model.named_modules():
        if need_to_prune(name, m, None):
            name_list.append(name)

    for name,m in model.named_modules():
        if need_to_prune(name, m, None):
            mask = mask_dict[name+'.weight_mask']
            weight = mask * m.weight
            weight = torch.sum(weight.abs(), [2, 3])
            for i in range(weight.shape[1]):
                start_name = name + '.{}'.format(i)
                graph.add_node(start_name)
                for j in range(weight.shape[0]):
                    try:
                        end_name = name_list[name_list.index(name) + 1] + '.{}'.format(j)
                        graph.add_node(end_name)
                        
                    except:
                        end_name = 'final.{}'.format(j)
                        graph.add_node(end_name)

                    graph.add_edge(start_name, end_name, weight=weight[j, i])
    
    edges_betweenness = edge_betweenness_centrality(graph, k=int(graph.number_of_nodes() / downsample))
    edges_betweenness = sorted((value,key) for (key,value) in edges_betweenness.items())
    for i in range(num_paths):
        edge = edges_betweenness[-i]
        kernel = '.'.join(edge[1][0].split(".")[:-1])
        start_index = int(edge[1][0].split(".")[-1])
        end_index = int(edge[1][1].split(".")[-1])
        mask = new_mask_dict[kernel + '.weight_mask']
        mask[end_index, start_index, :, :] = 0
        new_mask_dict[kernel + '.weight_mask'] = mask
        
    return new_mask_dict

def get_reverse_flatten_params_fun(params,get_count=False):
    """
    Returns a function which reshapes the flattened vector to its original hessian_shape
    if get_count=True it returs total number of elements for the non-trivial(iterator) case
    """
    if isinstance(params,nn.Parameter):
        def resize_param_fun(flatten_params):
            return flatten_params.view(params.size())
        return resize_param_fun
    else:
        list_of_sizes = []
        def resize_param_fun(flatten_params):
            c_sum = 0
            for numel,size in list_of_sizes:
                yield flatten_params[c_sum:c_sum+numel].view(size)
                c_sum += numel

        if get_count:
            total_element_number = 0
            for p in params:
                total_element_number += p.nelement()
                list_of_sizes.append((p.nelement(),p.size()))
            return resize_param_fun,total_element_number
        else:
            for p in params:
                list_of_sizes.append((p.nelement(),p.size()))
            return resize_param_fun

def flatten_params(params):
    """
    gets a iterator of Parameter/Variable/Tensor
    returns: [0] returns flatten(1d) version with length N
             [1] a generator function which accepts a Parameter/Variable/Tensor of length N
             and returns a generator of Parameter/Variable/Tensor with same sizes in order as the params.
    """
    if isinstance(params,nn.Parameter):
        return params.contiguous().view(-1)
    else:
        list_of_params = []
        for p in params:
            list_of_params.append(p.contiguous().view(-1))
        return torch.cat(list_of_params)
def hessian_vector_product(loss,params,vector,params_grad=None,retain_graph=False,flattened=False):
    """
    params: Case 1: Parameter
                Then the param:vector should be a Tensor with same size. The result is same size as the Parameter.
            Case 2: iterator of Parameters
                This is allowed only when flattened=True.
    loss: needed only params_grad is not provided
    vector: Same size as the params_grad. If you are flattened without providing the params_grad note that your vector
            match the size of the flattened parameters.
    params_grad: is for preventing recalculation and to be able to use in hessian
    flattened: if true then the params should be list of parameters. Then the hessian vector product is flattened.
        In this setting I am not returning the reverse functon that flatten_params generate since
        the only instance where I flatten is during the hessian and I get the same function during grad calcualtion.
        Future use cases may require and one can return.
    """

    params = list(params)

    
    params_grad = torch.autograd.grad(loss, params, create_graph=True)
    if flattened:
        params_grad = flatten_params(params_grad)
    else:
        params_grad = params_grad[0]
    if params_grad.is_cuda: vector= vector.cuda()
    # import pdb;pdb.set_trace()
    grad_vector_dot = torch.sum(params_grad * vector)
    hv_params = torch.autograd.grad(grad_vector_dot, params,retain_graph=retain_graph)
    if flattened:
        hv_params = flatten_params(hv_params)
    else:
        hv_params = hv_params[0]

    return hv_params.data

def prune_hessian_abs(model, mask_dict, num_paths, args):
    new_mask_dict = copy.deepcopy(mask_dict)
    named_params = model.named_parameters()
    params = []
    for name, m in named_params:
        if name + '_mask' in mask_dict:
            params.append(m)

    rev_f, n_elements = get_reverse_flatten_params_fun(params,get_count=True)
    vector = flatten_params((-p.data.clone() for p in params))
    if args.dataset == 'cifar10':
        train_set_loader, _, _ = cifar10_dataloaders(batch_size=args.batch_size, data_dir =args.data)
    elif args.dataset == 'cifar100':
        train_set_loader, _, _ = cifar100_dataloaders(batch_size=args.batch_size, data_dir =args.data)
    else:
        raise NotImplementedError
    image, label = next(iter(train_set_loader))
    if True:
        image = image.cuda()
        label = label.cuda()
    output = model(image)
    loss = torch.nn.functional.cross_entropy(output, label)
    flat_hv = hessian_vector_product(loss,params,vector,retain_graph=True,flattened=True)
    hv = rev_f(flat_hv)
    result = [torch.mul(-(w.data),h).abs() for w,h in zip(params,hv)]
    result_dict = {}
    result_flatten = []
    for key, param in zip(mask_dict.keys(), result):
        param[mask_dict[key] == 0] = -np.inf
        result_flatten.append(param.view(-1))
    result_flatten = torch.cat(result_flatten, 0)
    threshold, _ = torch.kthvalue(result_flatten, result_flatten.numel() - num_paths)
    for key, param in zip(mask_dict.keys(), result):
        param[mask_dict[key] == 0] = -np.inf
        new_mask_dict[key][param > threshold] = 0
    return new_mask_dict





def prune_taylor1_abs(model, mask_dict, num_paths, args):
    new_mask_dict = copy.deepcopy(mask_dict)
    named_params = model.named_parameters()
    params = []
    for name, m in named_params:
        if name + '_mask' in mask_dict:
            params.append(m)

    rev_f, n_elements = get_reverse_flatten_params_fun(params,get_count=True)
    vector = flatten_params((-p.data.clone() for p in params))
    if args.dataset == 'cifar10':
        train_set_loader, _, _ = cifar10_dataloaders(batch_size=args.batch_size, data_dir =args.data)
    elif args.dataset == 'cifar100':
        train_set_loader, _, _ = cifar100_dataloaders(batch_size=args.batch_size, data_dir =args.data)
    else:
        raise NotImplementedError
    image, label = next(iter(train_set_loader))
    if True:
        image = image.cuda()
        label = label.cuda()
    output = model(image)
    loss = torch.nn.functional.cross_entropy(output, label)
    grads = torch.autograd.grad(loss,params,retain_graph=True)
    result = [abs(torch.mul(-(w.data),g.data)) for w,g in zip(params,grads)]
    result_dict = {}
    result_flatten = []
    for key, param in zip(mask_dict.keys(), result):
        param[mask_dict[key] == 0] = -np.inf
        result_flatten.append(param.view(-1))
    result_flatten = torch.cat(result_flatten, 0)
    threshold, _ = torch.kthvalue(result_flatten, result_flatten.numel() - num_paths)
    for key, param in zip(mask_dict.keys(), result):
        param[mask_dict[key] == 0] = -np.inf
        new_mask_dict[key][param > threshold] = 0
    return new_mask_dict

def prune_intgrads(model, mask_dict, num_paths, args):
    new_mask_dict = copy.deepcopy(mask_dict)
    named_params = model.named_parameters()
    params = []
    params_name = []
    for name, m in named_params:
        if name + '_mask' in mask_dict:
            params.append(m)
            params_name.append(name)

    if args.dataset == 'cifar10':
        train_set_loader, _, _ = cifar10_dataloaders(batch_size=args.batch_size, data_dir =args.data)
    elif args.dataset == 'cifar100':
        train_set_loader, _, _ = cifar100_dataloaders(batch_size=args.batch_size, data_dir =args.data)
    else:
        raise NotImplementedError
    image, label = next(iter(train_set_loader))
    if True:
        image = image.cuda()
        label = label.cuda()
    result = []
    for n, p in zip(params_name, params):
        grads = []
        for alpha in np.arange(0.01, 1.01, 0.01):
            p.data.mul_(alpha)
            output = model(image)
            #print(output)
            loss = torch.nn.functional.cross_entropy(output, label)

            grad = torch.autograd.grad(loss,p)
            grads.append(grad[0])
            p.data.div_(alpha)
        sums = torch.sum(torch.stack(grads), 0)
        print(sums.shape)
        result.append(torch.abs(torch.mul(p.data, 0.01 * sums)))
    
    result_dict = {}
    result_flatten = []
    for key, param in zip(mask_dict.keys(), result):
        param[mask_dict[key] == 0] = -np.inf
        result_flatten.append(param.view(-1))
    result_flatten = torch.cat(result_flatten, 0)
    threshold, _ = torch.kthvalue(result_flatten, result_flatten.numel() - num_paths)
    for key, param in zip(mask_dict.keys(), result):
        param[mask_dict[key] == 0] = -np.inf
        new_mask_dict[key][param > threshold] = 0
    return new_mask_dict

def prune_identity(model, mask_dict, num_paths, args):
    return mask_dict


def prune_random(model, mask_dict, num_paths, args):
    new_mask_dict = copy.deepcopy(mask_dict)
    for _ in range(num_paths):
        end_index = None
        for name,m in model.named_modules():
            if need_to_prune(name, m, None):
                mask = mask_dict[name+'.weight_mask']
                weight = m.weight * mask
                weight = torch.sum(weight.abs(), [2,3]).cpu().detach().numpy()
                if end_index is None:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                try:
                    prob = np.abs(weight[:, start_index]) > 0
                except IndexError:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = np.abs(weight[:, start_index]) > 0
                prob = prob / (prob.sum() + 1e-10)

                counter = 0
                while prob.sum() == 0:
                    start_index = np.random.randint(0, weight.shape[1] - 1)
                    prob = np.abs(weight[:, start_index]) > 0
                    prob = prob / (prob.sum() + 1e-10)
                    counter = counter + 1
                    
                    if counter > 200000:
                        prob = np.ones(prob.shape)
                        prob = prob / prob.sum()

                end_index = np.random.choice(np.arange(weight.shape[0]), 1,
                            p=np.array(prob))[0]
                new_mask_dict[name+'.weight_mask'][end_index, start_index, :, :] = 0
                start_index = end_index
    return new_mask_dict



def prune_omp(model, mask_dict, num_paths, args):
    new_mask_dict = copy.deepcopy(mask_dict)
    named_params = model.named_parameters()
    params = []
    for name, m in named_params:
        if name + '_mask' in mask_dict:
            params.append(m)

    rev_f, n_elements = get_reverse_flatten_params_fun(params,get_count=True)
    vector = flatten_params((-p.data.clone() for p in params))
    result = [w.data.abs() for w in params]
    result_dict = {}
    result_flatten = []
    for key, param in zip(mask_dict.keys(), result):
        param[mask_dict[key] == 0] = -np.inf
        result_flatten.append(param.view(-1))
    result_flatten = torch.cat(result_flatten, 0)
    threshold, _ = torch.kthvalue(result_flatten, result_flatten.numel() - num_paths)
    for key, param in zip(mask_dict.keys(), result):
        param[mask_dict[key] == 0] = -np.inf
        new_mask_dict[key][param > threshold] = 0
    return new_mask_dict