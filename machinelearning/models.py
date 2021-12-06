import nn
import numpy as np

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(self.get_weights(), x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        num = nn.as_scalar(self.run(x))
        return 1 if num >= 0 else -1


    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        mistake = True
        while mistake:
            mistake = False
            for x, y in dataset.iterate_once(1):
                prediction =  self.get_prediction(x)
                true_value = nn.as_scalar(y)
                if prediction != true_value:
                    # weights = weights + (direction * multiplier)
                    self.get_weights().update(x, true_value)
                    mistake = True
            

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here

        # parameter matrices
        # an ixh matrix, where i=dimension of input vectors x, h=hidden layer size
        # aka batch features x input features
        self.W1 = nn.Parameter(1, 100)
        self.W2 = nn.Parameter(100, 1)

        # parameter vectors to learn during gradient descent
        # size h vector
        # 1 x input features
        self.b1 = nn.Parameter(1, 100)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        # f(x) = relu(x*W1 + b1) * W2 + b2
        innerRelu = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        relu = nn.ReLU(innerRelu)
        update2 = nn.Linear(relu, self.W2)
        result = nn.AddBias(update2, self.b2)
        return result

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        loss_check = float('inf')
        learning_rate = -.01
        while loss_check > .01:
            for x, y in dataset.iterate_once(10):
                loss = self.get_loss(x, y)
                grad_wrt_W1, grad_wrt_W2, grad_wrt_b1, grad_wrt_b2 = nn.gradients(loss, [self.W1, self.W2, self.b1, self.b2])
                self.W1.update(grad_wrt_W1, learning_rate)
                self.W2.update(grad_wrt_W2, learning_rate)
                self.b1.update(grad_wrt_b1, learning_rate)
                self.b2.update(grad_wrt_b2, learning_rate)
                loss_check = nn.as_scalar(self.get_loss(x, y))

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # parameter matrices
        # an ixh matrix, where i=dimension of input vectors x, h=hidden layer size
        # aka batch features x input features
        self.W1 = nn.Parameter(784, 500)
        self.W2 = nn.Parameter(500, 250)
        self.W3 = nn.Parameter(250, 10)

        # parameter vectors to learn during gradient descent
        # size h vector
        # 1 x input features
        self.b1 = nn.Parameter(1, 500)
        self.b2 = nn.Parameter(1, 250)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # f(x) = relu(relu(x*W1 + b1) * W2 + b2) * W3 + b3
        innerRelu1 = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        relu1 = nn.ReLU(innerRelu1)
        update1 = nn.Linear(relu1, self.W2)
        innerRelu2 = nn.AddBias(update1, self.b2)
        relu2 = nn.ReLU(innerRelu2)
        update2 = nn.Linear(relu2, self.W3)
        result = nn.AddBias(update2, self.b3)
        return result


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        accuracy = dataset.get_validation_accuracy()
        learning_rate = -.1
        while accuracy < .975:
            for x, y in dataset.iterate_once(300):
                loss = self.get_loss(x, y)
                grad_wrt_W1, grad_wrt_W2, grad_wrt_W3, grad_wrt_b1, grad_wrt_b2, grad_wrt_b3 = \
                    nn.gradients(loss, [self.W1, self.W2, self.W3, self.b1, self.b2, self.b3])
                self.W1.update(grad_wrt_W1, learning_rate)
                self.W2.update(grad_wrt_W2, learning_rate)
                self.W3.update(grad_wrt_W3, learning_rate)
                self.b1.update(grad_wrt_b1, learning_rate)
                self.b2.update(grad_wrt_b2, learning_rate)
                self.b3.update(grad_wrt_b3, learning_rate)
                accuracy = dataset.get_validation_accuracy()

class DeepQModel(object):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = -.4
        self.numTrainingGames = 3000
        # self.parameters = [nn.Parameter(self.state_size, 400), 
        #                 nn.Parameter(self.state_size, 250), 
        #                 nn.Parameter(self.state_size, 100)]
        self.batch_size = 100

        # parameter matrices
        # an ixh matrix, where i=dimension of input vectors x, h=hidden layer size
        # aka batch features x input features
        self.W1 = nn.Parameter(self.state_size, 200)
        self.W2 = nn.Parameter(200, 100)
        self.W3 = nn.Parameter(100, 30)
        self.W4 = nn.Parameter(30, self.num_actions)

        # parameter vectors to learn during gradient descent
        # size h vector
        # 1 x input features
        self.b1 = nn.Parameter(1, 200)
        self.b2 = nn.Parameter(1, 100)
        self.b3 = nn.Parameter(1, 30)
        self.b4 = nn.Parameter(1, self.num_actions)

        self.parameters = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.W4, self.b4]
        
        # self.W1 = nn.Parameter(self.state_size, 1000)
        # self.W2 = nn.Parameter(1000, self.num_actions)
        # self.b1 = nn.Parameter(1, 1000)
        # self.b2 = nn.Parameter(1, self.num_actions)

        # self.parameters = [self.W1, self.b1, self.W2, self.b2]


    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(states), Q_target)


    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a node with shape (batch_size x state_dim)
        Output:
            result: a node with shape (batch_size x num_actions) containing Q-value
                scores for each of the actions
        """
        "*** YOUR CODE HERE ***"
        # f(x) = relu(x*W1 + b1) * W2 + b2
        # innerRelu = nn.AddBias(nn.Linear(states, self.W1), self.b1)
        # relu = nn.ReLU(innerRelu)
        # update2 = nn.Linear(relu, self.W2)
        # result = nn.AddBias(update2, self.b2)
        # return result

        # f(x) = relu(relu(relu(x*W1 + b1) * W2 + b2) * W3 + b3) * W4 + b4
        innerRelu1 = nn.AddBias(nn.Linear(states, self.W1), self.b1)
        relu1 = nn.ReLU(innerRelu1)
        update1 = nn.Linear(relu1, self.W2)

        innerRelu2 = nn.AddBias(update1, self.b2)
        relu2 = nn.ReLU(innerRelu2)
        update2 = nn.Linear(relu2, self.W3)

        innerRelu3 = nn.AddBias(update2, self.b3)
        relu3 = nn.ReLU(innerRelu3)
        update3 = nn.Linear(relu3, self.W4)

        q_predict = nn.AddBias(update3, self.b4)
        
        # print(result.data)
        return q_predict

        # Q_value_scores = []
        # for s in states.data:
        #     # f(x) = relu(relu(x*W1 + b1) * W2 + b2) * W3 + b3
        #     innerRelu1 = nn.AddBias(nn.Linear(s, self.W1), self.b1)
        #     relu1 = nn.ReLU(innerRelu1)
        #     update1 = nn.Linear(relu1, self.W2)
        #     innerRelu2 = nn.AddBias(update1, self.b2)
        #     relu2 = nn.ReLU(innerRelu2)
        #     update2 = nn.Linear(relu2, self.W3)
        #     result = nn.AddBias(update2, self.b3)
        #     print(result)
        #     Q_value_scores.append(result)
        # return nn.Node(Q_value_scores)

    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        gradients = nn.gradients(self.get_loss(states, Q_target), self.parameters)
        for param, gradient in zip(self.parameters, gradients):
            param.update(gradient, self.learning_rate)




