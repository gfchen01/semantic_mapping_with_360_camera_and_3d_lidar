import semantic_mapping.utils as utils
import yaml

def test_meta_class_extraction():
    label = 'sofa long sofa'
    
    class_template_file = './config/objects_cic.yaml'
    with open(class_template_file, 'r') as file:
        class_template = yaml.load(file, Loader=yaml.FullLoader)

    extracted = utils.extract_meta_class(label, class_template)
    print(extracted)

if __name__ == "__main__":
    test_meta_class_extraction()