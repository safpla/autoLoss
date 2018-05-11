from models import gan
from models import gan_grid
import os
import utils

if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, 'config/' + 'gan.cfg')
    config = utils.Parser(config_path)
    config.print_config()
    model = gan.Gan(config, exp_name='dcgan_baseline_mnist_10')
    model.initialize_weights()

    # ----Load pretrain.----
    #load_dir = os.path.join(config.model_dir, 'dcgan_pretrain')
    #model.load_model(load_dir)

    model.train(save_model=True)
    print(model.get_inception_score(5000))
