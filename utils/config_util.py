from configparser import ConfigParser, ExtendedInterpolation
import json
import os

class Parser(object):
    def __init__(self, config_path):
        assert os.path.exists(config_path), '{} not exists.'.format(config_path)
        self.config = ConfigParser(
            delimiters='=',
            interpolation=ExtendedInterpolation())
        self.config.read(config_path)
        self.lambda1_stud = self.config.getfloat('stud', 'lambda1_stud')
        self.lambda2_stud = self.config.getfloat('stud', 'lambda2_stud')

    @property
    def num_pre_loss(self):
        return self.config.getint('rl', 'num_pre_loss')

    @property
    def dim_state_rl(self):
        return self.config.getint('rl', 'dim_state_rl')

    @property
    def dim_hidden_rl(self):
        return self.config.getint('rl', 'dim_hidden_rl')

    @property
    def dim_action_rl(self):
        return self.config.getint('rl', 'dim_action_rl')

    @property
    def lr_rl(self):
        return self.config.getfloat('rl', 'lr_rl')

    @property
    def reward_c(self):
        return self.config.getfloat('rl', 'reward_c')

    @property
    def explore_rate_decay_rl(self):
        return self.config.getint('rl', 'explore_rate_decay_rl')

    @property
    def explore_rate_rl(self):
        return self.config.getfloat('rl', 'explore_rate_rl')

    @property
    def total_episodes(self):
        return self.config.getint('rl', 'total_episodes')

    @property
    def max_training_step(self):
        return self.config.getint('rl', 'max_training_step')

    @property
    def update_frequency(self):
        return self.config.getint('rl', 'update_frequency')

    @property
    def exp_dir(self):
        return os.path.expanduser(self.config.get('env', 'exp_dir'))

    @property
    def data_dir(self):
        return os.path.expanduser(self.config.get('env', 'data_dir'))

    @property
    def model_dir(self):
        return os.path.expanduser(self.config.get('env', 'model_dir'))

    @property
    def student_model_name(self):
        return self.config.get('stud', 'student_model_name')

    @property
    def train_data_file(self):
        train_data_file = self.config.get('data', 'train_data_file')
        return os.path.join(self.data_dir, train_data_file)

    @property
    def valid_data_file(self):
        valid_data_file = self.config.get('data', 'valid_data_file')
        return os.path.join(self.data_dir, valid_data_file)

    @property
    def test_data_file(self):
        test_data_file = self.config.get('data', 'test_data_file')
        return os.path.join(self.data_dir, test_data_file)

    @property
    def num_sample_train(self):
        return self.config.getint('data', 'num_sample_train')

    @property
    def num_sample_valid(self):
         return self.config.getint('data', 'num_sample_valid')

    @property
    def batch_size(self):
        return self.config.getint('stud', 'batch_size')

    @property
    def dim_input_stud(self):
        return self.config.getint('stud', 'dim_input_stud')

    @property
    def dim_hidden_stud(self):
        return self.config.getint('stud', 'dim_hidden_stud')

    @property
    def dim_output_stud(self):
        return self.config.getint('stud', 'dim_output_stud')

    @property
    def lr_stud(self):
        return self.config.getfloat('stud', 'lr_stud')

    @property
    def valid_frequence_stud(self):
        return self.config.getint('stud', 'valid_frequence_stud')

    @property
    def max_endurance_stud(self):
        return self.config.getint('stud', 'max_endurance_stud')

    @property
    def reward_baseline_decay(self):
        return self.config.getfloat('rl', 'reward_baseline_decay')

    @property
    def timedelay_num(self):
        return self.config.getint('train', 'timedelay_num')

    @property
    def max_step(self):
        return self.config.getint('train', 'max_step')

    @property
    def summary_steps(self):
        return self.config.getint('train', 'summary_steps')

    @property
    def lr_policy_params(self):
        params = self.config.get('train', 'lr_policy_params', fallback='{}')
        return json.loads(params)


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    config_path = os.path.join(root_path, 'config/regression.cfg')
    config = Parser(config_path)
    print(config.exp_dir)
    print(config.model_dir)
