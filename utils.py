def get_filename(args, args_subset=None):
    args_dict = vars(args)
    filename = ''
    for ele in args_dict:
        if args_subset is not None and ele not in args_subset:
            continue
        filename += ele + '-' + str(args_dict[ele]) + '__'
    
    return filename[:-2]