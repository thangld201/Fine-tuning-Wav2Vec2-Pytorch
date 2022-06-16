import json

def dump_json(test_data,output_json_name=None):
    if output_json_name is not None:
            with open(output_json_name, "w",encoding='utf8') as output_file:
                json.dump(test_data,output_file,ensure_ascii=False)

def get_json(path):
    with open(path,"r", encoding='utf8') as f:
        data=json.load(f)
    return data