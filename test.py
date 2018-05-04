from models import gan
import os
import utils

if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, 'config/' + 'gan.cfg')
    config = utils.Parser(config_path)
    model = gan.Gan(config, exp_name='dcgan_pretrain')
    load_dir = os.path.join(config.model_dir, 'dcgan_pretrain')
    model.initialize_weights()
    #model.load_model(load_dir)
    model.train(save_model=True)
