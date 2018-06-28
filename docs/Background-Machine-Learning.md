# Background: Machine Learning

Given that a number of users of ML-Agents might not have a formal machine 
learning background, this page provides an overview to facilitate the 
understanding of ML-Agents. However, We will not attempt to provide a thorough 
treatment of machine learning as there are fantastic resources online.

考虑到：ML-Agents的一些用户可能没有正规的机器学习背景，为了促进ML-Agents的理解这页提供了一个
概述。

然而，因为网上有极好的资源，我们就不打算尝试提供机器学习彻底的论述。

Machine learning, a branch of artificial intelligence, focuses on learning 
patterns from data. The three main classes of machine learning algorithms
include: unsupervised learning, supervised learning and reinforcement learning. 
Each class of algorithm learns from a different type of data. The following 
paragraphs provide an overview for each of these classes of machine learning, 
as well as introductory examples.

机器学习，是人工智能的分支，着重于从数据中习得模式。

三种主要的机器学习算法包括：无监督学习，监督学习和强化学习。

每一类不同的算法从不同类型的数据中学习。

下面的段落为机器学习的每一个种类提供了概述，还提供介绍性例子。



## Unsupervised Learning

The goal of 
[unsupervised learning](https://en.wikipedia.org/wiki/Unsupervised_learning) is to group or cluster similar items in a 
data set. For example, consider the players of a game. We may want to group 
the players depending on how engaged they are with the game. This would enable
us to target different groups (e.g. for highly-engaged players we might
invite them to be beta testers for new features, while for unengaged players
we might email them helpful tutorials). Say that we wish to split our players 
into two groups. We would first define basic attributes of the players, such 
as the number of hours played, total money spent on in-app purchases and
number of levels completed. We can then feed this data set (three attributes 
for every player) to an unsupervised learning algorithm where we specify the 
number of groups to be two. The algorithm would then split the data set of
players into two groups where the players within each group would be similar
to each other.  Given the attributes we used to describe each player, in this
case, the output would be a split of all the players into two groups, where 
one group would semantically represent the engaged players and the second
group would semantically represent the unengaged players.

无监监督学习的目的是将数据集中的相似项分类或者聚类。

举个例子，考虑游戏中的玩家。

根据他们在游戏中的投入程度，我们可能想将玩家分类。这将使我们能标出不同的组。

比如说，我们想要将玩家分进两个组。

首先，我们要明确玩家们的基本属性，如：游戏时长，游戏总花费和通关数量。
然后，我们为用来分组的无监督学习算法，提供包含这些属性的数据集。
然后，算法将玩家的数据集分成两组，每组中项相似。

鉴于我们用来描述玩家的属性，在本例中，所有玩家将被分为两组，一组代表高投入玩家，一组代表低投入玩家。

With unsupervised learning, we did not provide specific examples of which
players are considered engaged and which are considered unengaged. We just
defined the appropriate attributes and relied on the algorithm to uncover
the two groups on its own. This type of data set is typically called an 
unlabeled data set as it is lacking these direct labels. Consequently, 
unsupervised learning can be helpful in situations where these labels can be
expensive or hard to produce. In the next paragraph, we overview supervised 
learning algorithms which accept input labels in addition to attributes.

在无监督学习中，我们不明确提供哪一个玩家可以被认为是高投入玩家，哪一个玩家可以被认为是
低频率玩家的例子。

我们只定义合适的属性和依赖算法本身来发现两个分组。

因为缺少直接的标签，这种类型的数据集被称为无标签数据集。

所以，无监督学习在标签昂贵和产生困难的场景中非常有用。

在下一段中，我们将概述监督学习，它除了加入属性，还接受标签输入。



## Supervised Learning

In [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning),
we do not want to just group similar items but directly
learn a mapping from each item to the group (or class) that it belongs to.
Returning to our earlier example of
clustering players, let's say we now wish to predict which of our players are
about to churn (that is stop playing the game for the next 30 days). We 
can look into our historical records and create a data set that
contains attributes of our players in addition to a label indicating whether
they have churned or not. Note that the player attributes we use for this
churn prediction task may be different from the ones we used for our earlier
clustering task. We can then feed this data set (attributes **and** label for
each player) into a supervised learning algorithm which would learn a mapping 
from the player attributes to a label indicating whether that player 
will churn or not. The intuition is that the supervised learning algorithm
will learn which values of these attributes typically correspond to players
who have churned and not churned (for example, it may learn that players
who spend very little and play for very short periods will most likely churn).
Now given this learned model, we can provide it the attributes of a
new player (one that recently started playing the game) and it would output
a _predicted_ label for that player. This prediction is the algorithms
expectation of whether the player will churn or not.We can now use these predictions to target the players
who are expected to churn and entice them to continue playing the game.

在监督学习中，我们并不想只将相似项分类，还想直接习得每一项到它所属组的映射。

返回我们较早的玩家聚类的例子，比如说我们现在想要预测玩家中谁将停止游戏。

我们可以查找我们的历史记录，并且并且创建一个包含玩家属性和表明玩家是否会停止游戏标签的数据集。

注意：我们用于预测玩家是停止游戏的属性，可能与之前聚类任务的不同。

然后，我们将数据集提供给监督学习算法，其将产生一个从玩家属性到标签指示结果的映射。

直接的理解是：监督学习算法将习得属性中哪一个值与将要停止游戏的玩家相关。

现在给出这个学习模型，我们可以提供一个新玩家的属性，它可以产生一个标签预测。

这个预测是玩家是否会停止游戏的算法期望。

现在，我们可以使用这个预测，来标出可能会停止游戏的玩家，并激励他们继续游戏。


As you may have noticed, for both supervised and unsupervised learning, there
are two tasks that need to be performed: attribute selection and model
selection. Attribute selection (also called feature selection) pertains to
selecting how we wish to represent the entity of interest, in this case, the
player. Model selection, on the other hand, pertains to selecting the
algorithm (and its parameters) that perform the task well. Both of these
tasks are active areas of machine learning research and, in practice, require
several iterations to achieve good performance.

正如你可能注意到的，监督学习和无监督学都需要完成两个任务：属性选择和模型选择。

属性选择是关于选择如何表示利益实体，本例中，为玩家。

另一方面，模型选择是关于选择算法和参数，使任务表现良好。

这两个任务都是机器学习的活跃领域，在实践中，需要几次迭代来达到最好的性能。

We now switch to reinforcement learning, the third class of
machine learning algorithms, and arguably the one most relevant for ML-Agents.

现在，我们转向强化学习，第三种机器学习算法，ML-Agents最相关的算法。


## Reinforcement Learning

[Reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
can be viewed as a form of learning for sequential
decision making that is commonly associated with controlling robots (but is,
in fact, much more general). Consider an autonomous firefighting robot that is
tasked with navigating into an area, finding the fire and neutralizing it. At
any given moment, the robot perceives the environment through its sensors (e.g.
camera, heat, touch), processes this information and produces an action (e.g.
move to the left, rotate the water hose, turn on the water). In other words,
it is continuously making decisions about how to interact in this environment
given its view of the world (i.e. sensors input) and objective (i.e.
neutralizing the fire). Teaching a robot to be a successful firefighting
machine is precisely what reinforcement learning is designed to do. 

More specifically, the goal of reinforcement learning is to learn a **policy**, 
which is essentially a mapping from **observations** to **actions**. An 
observation is what the robot can measure from its **environment** (in this 
case, all its sensory inputs) and an action, in its most raw form, is a change
to the configuration of the robot (e.g. position of its base, position of
its water hose and whether the hose is on or off). 

The last remaining piece
of the reinforcement learning task is the **reward signal**. When training a
robot to be a mean firefighting machine, we provide it with rewards (positive 
and negative) indicating how well it is doing on completing the task.
Note that the robot does not _know_ how to put out fires before it is trained. 
It learns the objective because it receives a large positive reward when it puts 
out the fire and a small negative reward for every passing second. The fact that 
rewards are sparse (i.e. may not be provided at every step, but only when a 
robot arrives at a success or failure situation), is a defining characteristic of 
reinforcement learning and precisely why learning good policies can be difficult 
(and/or time-consuming) for complex environments. 

<p align="center">
  <img src="images/rl_cycle.png" alt="The reinforcement learning cycle."/>
</p>

[Learning a policy](https://blogs.unity3d.com/2017/08/22/unity-ai-reinforcement-learning-with-q-learning/)
usually requires many trials and iterative
policy updates. More specifically, the robot is placed in several
fire situations and over time learns an optimal policy which allows it
to put our fires more effectively. Obviously, we cannot expect to train a
robot repeatedly in the real world, particularly when fires are involved. This
is precisely why the use of 
[Unity as a simulator](https://blogs.unity3d.com/2018/01/23/designing-safer-cities-through-simulations/)
serves as the perfect training grounds for learning such behaviors.
While our discussion of reinforcement learning has centered around robots,
there are strong parallels between robots and characters in a game. In fact,
in many ways, one can view a non-playable character (NPC) as a virtual
robot, with its own observations about the environment, its own set of actions
and a specific objective. Thus it is natural to explore how we can
train behaviors within Unity using reinforcement learning. This is precisely
what ML-Agents offers. The video linked below includes a reinforcement
learning demo showcasing training character behaviors using ML-Agents.

<p align="center">
    <a href="http://www.youtube.com/watch?feature=player_embedded&v=fiQsmdwEGT8" target="_blank">
        <img src="http://img.youtube.com/vi/fiQsmdwEGT8/0.jpg" alt="RL Demo" width="400" border="10" />
    </a>
</p>

Similar to both unsupervised and supervised learning, reinforcement learning
also involves two tasks: attribute selection and model selection.
Attribute selection is defining the set of observations for the robot
that best help it complete its objective, while model selection is defining
the form of the policy (mapping from observations to actions) and its
parameters. In practice, training behaviors is an iterative process that may
require changing the attribute and model choices.

## Training and Inference

One common aspect of all three branches of machine learning is that they
all involve a **training phase** and an **inference phase**. While the
details of the training and inference phases are different for each of the
three, at a high-level, the training phase involves building a model
using the provided data, while the inference phase involves applying this
model to new, previously unseen, data. More specifically:
* For our unsupervised learning
example, the training phase learns the optimal two clusters based 
on the data describing existing players, while the inference phase assigns a 
new player to one of these two clusters. 
* For our supervised learning example, the 
training phase learns the mapping from player attributes to player label
(whether they churned or not), and the inference phase predicts whether 
a new player will churn or not based on that learned mapping. 
* For our reinforcement learning example, the training phase learns the
optimal policy through guided trials, and in the inference phase, the agent
observes and tales actions in the wild using its learned policy.

To briefly summarize: all three classes of algorithms involve training
and inference phases in addition to attribute and model selections. What
ultimately separates them is the type of data available to learn from. In
unsupervised learning our data set was a collection of attributes, in
supervised learning our data set was a collection of attribute-label pairs, 
and, lastly, in reinforcement learning our data set was a collection of 
observation-action-reward tuples.

## Deep Learning

[Deep learning](https://en.wikipedia.org/wiki/Deep_learning) is a family of 
algorithms that can be used to address any of the problems introduced 
above. More specifically, they can be used to solve both attribute and 
model selection tasks. Deep learning has gained popularity in recent 
years due to its outstanding performance on several challenging machine learning 
tasks. One example is [AlphaGo](https://en.wikipedia.org/wiki/AlphaGo), 
a  [computer Go](https://en.wikipedia.org/wiki/Computer_Go) program, that 
leverages deep learning, that was able to beat Lee Sedol (a Go world champion).

A key characteristic of deep learning algorithms is their ability learn very
complex functions from large amounts of training data. This makes them a
natural choice for reinforcement learning tasks when a large amount of data
can be generated, say through the use of a simulator or engine such as Unity.
By generating hundreds of thousands of simulations of
the environment within Unity, we can learn policies for very complex environments
(a complex environment is one where the number of observations an agent perceives
and the number of actions they can take are large).
Many of the algorithms we provide in ML-Agents use some form of deep learning,
built on top of the open-source library, [TensorFlow](Background-TensorFlow.md).
