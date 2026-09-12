"""
The file contains utility functions for the simulations.
"""

#import torch
import tensorflow as tf

def DataGen(args, SysModel_data, fileName):

    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatch(args, args.N_E, args.T, randomInit=args.randomInit_train)
    train_input = SysModel_data.Input
    train_target = SysModel_data.Target
    ### init conditions ###
    train_init = SysModel_data.m1x_0_batch #size: N_E x m x 1
    ### length mask ###
    if args.randomLength:
        train_lengthMask = SysModel_data.lengthMask

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatch(args, args.N_CV, args.T, randomInit=args.randomInit_cv)
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target
    cv_init = SysModel_data.m1x_0_batch #size: N_CV x m x 1
    ### length mask ###
    if args.randomLength:
        cv_lengthMask = SysModel_data.lengthMask

    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatch(args, args.N_T, args.T_test, randomInit=args.randomInit_test)
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target
    test_init = SysModel_data.m1x_0_batch #size: N_T x m x 1
    ### length mask ###
    if args.randomLength:
        test_lengthMask = SysModel_data.lengthMask

    #################
    ### Save Data ###
    #################
    #if(args.randomLength):
    #    torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init, train_lengthMask,cv_lengthMask,test_lengthMask], fileName)
    #else:
    #    torch.save([train_input, train_target, cv_input, cv_target, test_input, test_target,train_init, cv_init, test_init], fileName)
    if args.randomLength:
        data_dict = {
            "train_input": train_input, "train_target": train_target,
            "cv_input": cv_input, "cv_target": cv_target,
            "test_input": test_input, "test_target": test_target,
            "train_init": train_init, "cv_init": cv_init, "test_init": test_init,
            "train_lengthMask": train_lengthMask, "cv_lengthMask": cv_lengthMask, "test_lengthMask": test_lengthMask
        }
    else:
        data_dict = {
            "train_input": train_input, "train_target": train_target,
            "cv_input": cv_input, "cv_target": cv_target,
            "test_input": test_input, "test_target": test_target,
            "train_init": train_init, "cv_init": cv_init, "test_init": test_init
        }
    dataset = tf.data.Dataset.from_tensors(data_dict)
    tf.data.Dataset.save(dataset, fileName)

    ## Loading the dataset - TensorFlow
    #loaded_dataset = tf.data.Dataset.load(fileName)
    #for data in loaded_dataset:
    #    current_train_input = data["train_input"]
    #    if "train_lengthMask" in data:
    #        current_train_lengthMask = data["train_lengthMask"]

def DecimateData(all_tensors, t_gen,t_mod, offset=0):
    
    # ratio: defines the relation between the sampling time of the true process and of the model (has to be an integer)
    ratio = round(t_mod / t_gen)

    i = 0
    all_tensors_out = all_tensors
    for tensor in all_tensors:
        tensor = tensor[:, (0 + offset)::ratio]
        if i == 0:
            #all_tensors_out = torch.cat([tensor], dim=0).view(1, all_tensors.size()[1], -1)
            all_tensors_out = tf.reshape(tf.concat([tensor], axis=0), [1, tf.shape(all_tensors)[1], -1])
        else:
            #all_tensors_out = torch.cat([all_tensors_out, tensor.view(1, all_tensors.size()[1], -1)], dim=0)
            all_tensors_out = tf.concat([all_tensors_out, tf.reshape(tensor, [1, tf.shape(all_tensors)[1], -1])], axis=0)
        i += 1

    return all_tensors_out

def Decimate_and_perturbate_Data(true_process, delta_t, delta_t_mod, N_examples, h, lambda_r, offset=0):
    
    # Decimate high resolution process
    decimated_process = DecimateData(true_process, delta_t, delta_t_mod, offset)

    noise_free_obs = getObs(decimated_process,h)

    # Replicate for computation purposes
    #decimated_process = torch.cat(int(N_examples)*[decimated_process])
    #noise_free_obs = torch.cat(int(N_examples)*[noise_free_obs])
    decimated_process = tf.concat(int(N_examples) * [decimated_process], axis=0)
    noise_free_obs = tf.concat(int(N_examples) * [noise_free_obs], axis=0)

    # Observations; additive Gaussian Noise
    #observations = noise_free_obs + torch.randn_like(decimated_process) * lambda_r
    observations = noise_free_obs + tf.random.normal(shape=tf.shape(decimated_process)) * lambda_r
    return [decimated_process, observations]

def getObs(sequences, h):
    i = 0
    #sequences_out = torch.zeros_like(sequences)
    sequences_out = tf.zeros_like(sequences)
    for sequence in sequences:
        for t in range(sequence.size()[1]):
            sequences_out[i, :, t] = h(sequence[:, t])
    i = i + 1

    return sequences_out

def Short_Traj_Split(data_target, data_input, T):### Random Init is automatically incorporated
    #data_target = list(torch.split(data_target,T+1,2)) # +1 to reserve for init
    #data_input = list(torch.split(data_input,T+1,2)) # +1 to reserve for init
    
    #total_len_target = tf.shape(data_target)[2]
    #total_len_input = tf.shape(data_input)[2]
    #chunk_size = T + 1
    #sizes_target = [chunk_size] * int(total_len_target // chunk_size)
    #if total_len_target % chunk_size > 0:
    #    sizes_target.append(int(total_len_target % chunk_size))
    #sizes_input = [chunk_size] * int(total_len_input // chunk_size)
    #if total_len_input % chunk_size > 0:
    #    sizes_input.append(int(total_len_input % chunk_size))
    #data_target = tf.split(data_target, num_or_size_splits=sizes_target, axis=2)
    #data_input = tf.split(data_input, num_or_size_splits=sizes_input, axis=2)
    data_target = tf.split(data_target, num_or_size_splits=T + 1, axis=2)
    data_input = tf.split(data_input, num_or_size_splits=T + 1, axis=2)
    
    data_target.pop() # Remove the last one which may not fullfill length T
    data_input.pop() # Remove the last one which may not fullfill length T

    #data_target = torch.squeeze(torch.cat(list(data_target), dim=0))#Back to tensor and concat together
    #data_input = torch.squeeze(torch.cat(list(data_input), dim=0))#Back to tensor and concat together
    data_target = tf.squeeze(tf.concat(data_target, axis=0))
    data_input = tf.squeeze(tf.concat(data_input, axis=0))
    # Split out init
    target = data_target[:, :, 1:]
    input = data_input[:, :, 1:]
    init = data_target[:, :, 0]
    return [target, input, init]
