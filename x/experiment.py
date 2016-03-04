import cPickle
import numpy as np
from agnez.video import make_gif


class Experiment(object):
    """Experiment Base Class

    """
    def __init__(self, environment, agent, seed=123):
        self.environment = environment
        self.agent = agent
        self.seed = seed
        np.random.seed(seed)

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def test(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError


class Qlearn(Experiment):
    """Q-learning trainer

    """
    def __init__(self, environment, agent, seed=123):
        super(Qlearn, self).__init__(environment, agent)

    def build(self, optimizer="sgd", loss="mse",
              experience=None):
        self.agent.build(optimizer, loss, experience)

    def train(self, epoch=1, batch_size=1, exp_batch_size=0,
              epsilon=0.1, gamma=0.9,
              reset_memory=True, verbose=0):
        """Q-learning trainer

        Adapt model to minimize:
        (t - Q(s, a))^2 , where
        t = r + \gamma * argmax_a' Q(s', a')
        """
        num_actions = self.agent.num_actions

        if reset_memory:
            self.agent.reset()
        for e in xrange(epoch):
            self.environment.reset()
            game_over = False
            loss = 0
            # get initial observation, start game
            obs_t = self.environment.observe()
            while not game_over:
                obs_tm1 = obs_t

                # get next action
                if np.random.rand() <= epsilon:  # take random action
                    action = np.random.randint(0, num_actions, size=1)
                else:
                    action = self.agent.act(obs_tm1)

                # apply action, get rewareds and new state
                obs_t, reward, game_over = self.environment.update(action)

                # store experience
                self.agent.remember(obs_tm1, action, reward, obs_t, game_over)

                # adapt model
                loss += self.agent.update(batch_size=batch_size,
                                          exp_batch_size=exp_batch_size,
                                          gamma=gamma)
            if verbose > 1:
                win = "win" if reward == 1 else "lose"
                print(e, loss, win)

    def test(self, epoch=1, batch_size=1, verbose=0,
             visualize=None):
        frames = np.zeros((0, ) + self.agent.input_shape)
        frames = frames.transpose(0, 2, 3, 1)

        for e in xrange(epoch):
            self.environment.reset()
            game_over = False
            loss = 0
            # get initial observation, start game
            obs_t = self.environment.observe()
            while not game_over:
                obs_tm1 = obs_t

                # get next action
                action = self.agent.act(obs_tm1)

                # apply action, get rewareds and new state
                obs_t, reward, game_over = self.environment.update(action)

                frames = np.concatenate([frames, obs_t.transpose(0, 2, 3, 1)],
                                        axis=0)
            if verbose > 1:
                win = "win" if reward == 1 else "lose"
                print(e, loss, win)
        if visualize:
            frames = np.repeat(frames, 3, axis=-1)
            make_gif(frames[:-visualize['n_frames']],
                     filepath=visualize['filepath'], gray=visualize['gray'])

    def save(self, filepath):
        self.agent.model.save_weights(filepath + ".h5")
        cPickle.dump(self.agnent.meomory.memory, file(filepath+".pkl"), -1)
