import yaml

def read_yaml_file(file_name):
    config = None
    with open(file_name, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as err:
            print(err)
    print(config)
    return config
