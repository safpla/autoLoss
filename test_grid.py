from models import gan
from models import gan_grid
import os
import utils

if __name__ == '__main__':
    root_path = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(root_path, 'config/' + 'gan_grid.cfg')
    config = utils.Parser(config_path)
    model = gan_grid.Gan_grid(config, exp_name='gan_grid_baseline')
    model.initialize_weights()
    #model.load_model(load_dir)
    #for i in range(5):
    #    print(model.get_inception_score(100))
    model.train(save_model=True)
    #model.train()
