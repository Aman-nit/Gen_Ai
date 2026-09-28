from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

load_dotenv()  # Load environment variables from .env file

# DEFINING THE LLM
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3-0324",
    temperature=0.7,
    task="text-generation")

model1 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate.from_template(
    template = 'Genrate a short but simpler notes from the following text \n {text}'
)

prompt2 = PromptTemplate.from_template(
    template = 'Genrate 5 simple questions answer like Quiz from the following text \n {text}'
)


prompt3 = PromptTemplate.from_template(
    template = 'merge the provided notes and quiz into a single documents \n Notes: {notes} \n Quiz: {quiz}'
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'notes':prompt1 | model1 | parser,
        'quiz':prompt2 | model1 | parser
    }
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
CS229 Lecture notes
Andrew Ng
Part V
Support Vector Machines
This set of notes presents the Support Vector Machine (SVM) learning algorithm. SVMs are among the best (and many believe is indeed the best)
“off-the-shelf” supervised learning algorithm. To tell the SVM story, we’ll
need to first talk about margins and the idea of separating data with a large
“gap.” Next, we’ll talk about the optimal margin classifier, which will lead
us into a digression on Lagrange duality. We’ll also see kernels, which give
a way to apply SVMs efficiently in very high dimensional (such as infinitedimensional) feature spaces, and finally, we’ll close off the story with the
SMO algorithm, which gives an efficient implementation of SVMs.
1 Margins: Intuition
We’ll start our story on SVMs by talking about margins. This section will
give the intuitions about margins and about the “confidence” of our predictions; these ideas will be made formal in Section 3.
Consider logistic regression, where the probability p(y = 1|x; θ) is modeled by hθ(x) = g(θ
T x). We would then predict “1” on an input x if and
only if hθ(x) ≥ 0.5, or equivalently, if and only if θ
T x ≥ 0. Consider a
positive training example (y = 1). The larger θ
T x is, the larger also is
hθ(x) = p(y = 1|x; w, b), and thus also the higher our degree of “confidence”
that the label is 1. Thus, informally we can think of our prediction as being
a very confident one that y = 1 if θ
T x  0. Similarly, we think of logistic
regression as making a very confident prediction of y = 0, if θ
T x  0. Given
a training set, again informally it seems that we’d have found a good fit to
the training data if we can find θ so that θ
T x
(i)  0 whenever y
(i) = 1, and
1
2
θ
T x
(i)  0 whenever y
(i) = 0, since this would reflect a very confident (and
correct) set of classifications for all the training examples. This seems to be
a nice goal to aim for, and we’ll soon formalize this idea using the notion of
functional margins.
For a different type of intuition, consider the following figure, in which x’s
represent positive training examples, o’s denote negative training examples,
a decision boundary (this is the line given by the equation θ
T x = 0, and
is also called the separating hyperplane) is also shown, and three points
have also been labeled A, B and C.
   B
A
C
Notice that the point A is very far from the decision boundary. If we are
asked to make a prediction for the value of y at at A, it seems we should be
quite confident that y = 1 there. Conversely, the point C is very close to
the decision boundary, and while it’s on the side of the decision boundary
on which we would predict y = 1, it seems likely that just a small change to
the decision boundary could easily have caused out prediction to be y = 0.
Hence, we’re much more confident about our prediction at A than at C. The
point B lies in-between these two cases, and more broadly, we see that if
a point is far from the separating hyperplane, then we may be significantly
more confident in our predictions. Again, informally we think it’d be nice if,
given a training set, we manage to find a decision boundary that allows us
to make all correct and confident (meaning far from the decision boundary)
predictions on the training examples. We’ll formalize this later using the
notion of geometric margins.
3
2 Notation
To make our discussion of SVMs easier, we’ll first need to introduce a new
notation for talking about classification. We will be considering a linear
classifier for a binary classification problem with labels y and features x.
From now, we’ll use y ∈ {−1, 1} (instead of {0, 1}) to denote the class labels.
Also, rather than parameterizing our linear classifier with the vector θ, we
will use parameters w, b, and write our classifier as
hw,b(x) = g(w
T x + b).
Here, g(z) = 1 if z ≥ 0, and g(z) = −1 otherwise. This “w, b” notation
allows us to explicitly treat the intercept term b separately from the other
parameters. (We also drop the convention we had previously of letting x0 = 1
be an extra coordinate in the input feature vector.) Thus, b takes the role of
what was previously θ0, and w takes the role of [θ1 . . . θn]
T
.
Note also that, from our definition of g above, our classifier will directly
predict either 1 or −1 (cf. the perceptron algorithm), without first going
through the intermediate step of estimating the probability of y being 1
(which was what logistic regression did).
3 Functional and geometric margins
Lets formalize the notions of the functional and geometric margins. Given a
training example (x
(i)
, y
(i)
), we define the functional margin of (w, b) with
respect to the training example
γˆ
(i) = y
(i)
(w
T x + b).
Note that if y
(i) = 1, then for the functional margin to be large (i.e., for our
prediction to be confident and correct), then we need w
T x + b to be a large
positive number. Conversely, if y
(i) = −1, then for the functional margin to
be large, then we need w
T x + b to be a large negative number. Moreover,
if y
(i)
(w
T x + b) > 0, then our prediction on this example is correct. (Check
this yourself.) Hence, a large functional margin represents a confident and a
correct prediction.
For a linear classifier with the choice of g given above (taking values in
{−1, 1}), there’s one property of the functional margin that makes it not a
very good measure of confidence, however. Given our choice of g, we note that
if we replace w with 2w and b with 2b, then since g(w
T x+b) = g(2w
T x+2b),
4
this would not change hw,b(x) at all. I.e., g, and hence also hw,b(x), depends
only on the sign, but not on the magnitude, of w
T x + b. However, replacing
(w, b) with (2w, 2b) also results in multiplying our functional margin by a
factor of 2. Thus, it seems that by exploiting our freedom to scale w and b,
we can make the functional margin arbitrarily large without really changing
anything meaningful. Intuitively, it might therefore make sense to impose
some sort of normalization condition such as that ||w||2 = 1; i.e., we might
replace (w, b) with (w/||w||2, b/||w||2), and instead consider the functional
margin of (w/||w||2, b/||w||2). We’ll come back to this later.
Given a training set S = {(x
(i)
, y
(i)
);i = 1, . . . , m}, we also define the
function margin of (w, b) with respect to S as the smallest of the functional
margins of the individual training examples. Denoted by γˆ, this can therefore
be written:
γˆ = min
i=1,...,m
γˆ
(i)
.
Next, lets talk about geometric margins. Consider the picture below:
A w
γ
B
(i)
The decision boundary corresponding to (w, b) is shown, along with the
vector w. Note that w is orthogonal (at 90◦
) to the separating hyperplane.
(You should convince yourself that this must be the case.) Consider the
point at A, which represents the input x
(i) of some training example with
label y
(i) = 1. Its distance to the decision boundary, γ
(i)
, is given by the line
segment AB.
How can we find the value of γ
(i)
? Well, w/||w|| is a unit-length vector
pointing in the same direction as w. Since A represents x
(i)
, we therefore
5
find that the point B is given by x
(i) − γ
(i)
· w/||w||. But this point lies on
the decision boundary, and all points x on the decision boundary satisfy the
equation w
T x + b = 0. Hence,
w
T

x
(i) − γ
(i) w
||w||
+ b = 0.
Solving for γ
(i) yields
γ
(i) =
w
T x
(i) + b
||w|| =

w
||w||T
x
(i) +
b
||w||.
This was worked out for the case of a positive training example at A in the
figure, where being on the “positive” side of the decision boundary is good.
More generally, we define the geometric margin of (w, b) with respect to a
training example (x
(i)
, y
(i)
) to be
γ
(i) = y
(i)
 
w
||w||T
x
(i) +
b
||w||!
.
Note that if ||w|| = 1, then the functional margin equals the geometric
margin—this thus gives us a way of relating these two different notions of
margin. Also, the geometric margin is invariant to rescaling of the parameters; i.e., if we replace w with 2w and b with 2b, then the geometric margin
does not change. This will in fact come in handy later. Specifically, because
of this invariance to the scaling of the parameters, when trying to fit w and b
to training data, we can impose an arbitrary scaling constraint on w without
changing anything important; for instance, we can demand that ||w|| = 1, or
|w1| = 5, or |w1 + b| + |w2| = 2, and any of these can be satisfied simply by
rescaling w and b.
Finally, given a training set S = {(x
(i)
, y
(i)
);i = 1, . . . , m}, we also define
the geometric margin of (w, b) with respect to S to be the smallest of the
geometric margins on the individual training examples:
γ = min
i=1,...,m
γ
(i)
.
4 The optimal margin classifier
Given a training set, it seems from our previous discussion that a natural
desideratum is to try to find a decision boundary that maximizes the (geometric) margin, since this would reflect a very confident set of predictions
6
on the training set and a good “fit” to the training data. Specifically, this
will result in a classifier that separates the positive and the negative training
examples with a “gap” (geometric margin).
For now, we will assume that we are given a training set that is linearly
separable; i.e., that it is possible to separate the positive and negative examples using some separating hyperplane. How we we find the one that
achieves the maximum geometric margin? We can pose the following optimization problem:
maxγ,w,b γ
s.t. y
(i)
(w
T x
(i) + b) ≥ γ, i = 1, . . . , m
||w|| = 1.
I.e., we want to maximize γ, subject to each training example having functional margin at least γ. The ||w|| = 1 constraint moreover ensures that the
functional margin equals to the geometric margin, so we are also guaranteed
that all the geometric margins are at least γ. Thus, solving this problem will
result in (w, b) with the largest possible geometric margin with respect to the
training set.
If we could solve the optimization problem above, we’d be done. But the
“||w|| = 1” constraint is a nasty (non-convex) one, and this problem certainly
isn’t in any format that we can plug into standard optimization software to
solve. So, lets try transforming the problem into a nicer one. Consider:
maxγ,w,b
γˆ
||w||
s.t. y
(i)
(w
T x
(i) + b) ≥ γˆ, i = 1, . . . , m
Here, we’re going to maximize γˆ/||w||, subject to the functional margins all
being at least γˆ. Since the geometric and functional margins are related by
γ = γˆ/||w|, this will give us the answer we want. Moreover, we’ve gotten rid
of the constraint ||w|| = 1 that we didn’t like. The downside is that we now
have a nasty (again, non-convex) objective
γˆ
||w|| function; and, we still don’t
have any off-the-shelf software that can solve this form of an optimization
problem.
Lets keep going. Recall our earlier discussion that we can add an arbitrary
scaling constraint on w and b without changing anything. This is the key idea
we’ll use now. We will introduce the scaling constraint that the functional
margin of w, b with respect to the training set must be 1:
γˆ = 1.
7
Since multiplying w and b by some constant results in the functional margin
being multiplied by that same constant, this is indeed a scaling constraint,
and can be satisfied by rescaling w, b. Plugging this into our problem above,
and noting that maximizing γˆ/||w|| = 1/||w|| is the same thing as minimizing
||w||2
, we now have the following optimization problem:
minγ,w,b
1
2
||w||2
s.t. y
(i)
(w
T x
(i) + b) ≥ 1, i = 1, . . . , m
We’ve now transformed the problem into a form that can be efficiently
solved. The above is an optimization problem with a convex quadratic objective and only linear constraints. Its solution gives us the optimal margin classifier. This optimization problem can be solved using commercial
quadratic programming (QP) code.1
While we could call the problem solved here, what we will instead do is
make a digression to talk about Lagrange duality. This will lead us to our
optimization problem’s dual form, which will play a key role in allowing us to
use kernels to get optimal margin classifiers to work efficiently in very high
dimensional spaces. The dual form will also allow us to derive an efficient
algorithm for solving the above optimization problem that will typically do
much better than generic QP software.
5 Lagrange duality
Lets temporarily put aside SVMs and maximum margin classifiers, and talk
about solving constrained optimization problems.
Consider a problem of the following form:
minw f(w)
s.t. hi(w) = 0, i = 1, . . . , l.
Some of you may recall how the method of Lagrange multipliers can be used
to solve it. (Don’t worry if you haven’t seen it before.) In this method, we
define the Lagrangian to be
L(w, β) = f(w) +
X
l
i=1
βihi(w)
1You may be familiar with linear programming, which solves optimization problems
that have linear objectives and linear constraints. QP software is also widely available,
which allows convex quadratic objectives and linear constraints.
8
Here, the βi
’s are called the Lagrange multipliers. We would then find
and set L’s partial derivatives to zero:
∂L
∂wi
= 0;
∂L
∂βi
= 0,
and solve for w and β.
In this section, we will generalize this to constrained optimization problems in which we may have inequality as well as equality constraints. Due to
time constraints, we won’t really be able to do the theory of Lagrange duality
justice in this class,2 but we will give the main ideas and results, which we
will then apply to our optimal margin classifier’s optimization problem.
Consider the following, which we’ll call the primal optimization problem:
minw f(w)
s.t. gi(w) ≤ 0, i = 1, . . . , k
hi(w) = 0, i = 1, . . . , l.
To solve it, we start by defining the generalized Lagrangian
L(w, α, β) = f(w) +
X
k
i=1
αigi(w) +
X
l
i=1
βihi(w).
Here, the αi
’s and βi
’s are the Lagrange multipliers. Consider the quantity
θP(w) = max
α,β : αi≥0
L(w, α, β).
Here, the “P” subscript stands for “primal.” Let some w be given. If w
violates any of the primal constraints (i.e., if either gi(w) > 0 or hi(w) 6= 0
for some i), then you should be able to verify that
θP(w) = max
α,β : αi≥0
f(w) +
X
k
i=1
αigi(w) +
X
l
i=1
βihi(w) (1)
= ∞. (2)
Conversely, if the constraints are indeed satisfied for a particular value of w,
then θP(w) = f(w). Hence,
θP(w) =

f(w) if w satisfies primal constraints
∞ otherwise.
2Readers interested in learning more about this topic are encouraged to read, e.g., R.
T. Rockarfeller (1970), Convex Analysis, Princeton University Press.
9
Thus, θP takes the same value as the objective in our problem for all values of w that satisfies the primal constraints, and is positive infinity if the
constraints are violated. Hence, if we consider the minimization problem
min
w
θP(w) = min
w
max
α,β : αi≥0
L(w, α, β),
we see that it is the same problem (i.e., and has the same solutions as) our
original, primal problem. For later use, we also define the optimal value of
the objective to be p
∗ = minw θP(w); we call this the value of the primal
problem.
Now, lets look at a slightly different problem. We define
θD(α, β) = min
w
L(w, α, β).
Here, the “D” subscript stands for “dual.” Note also that whereas in the
definition of θP we were optimizing (maximizing) with respect to α, β, here
are are minimizing with respect to w.
We can now pose the dual optimization problem:
max
α,β : αi≥0
θD(α, β) = max
α,β : αi≥0
min
w
L(w, α, β).
This is exactly the same as our primal problem shown above, except that the
order of the “max” and the “min” are now exchanged. We also define the
optimal value of the dual problem’s objective to be d
∗ = maxα,β : αi≥0 θD(w).
How are the primal and the dual problems related? It can easily be shown
that
d
∗ = max
α,β : αi≥0
min
w
L(w, α, β) ≤ min
w
max
α,β : αi≥0
L(w, α, β) = p
∗
.
(You should convince yourself of this; this follows from the “max min” of a
function always being less than or equal to the “min max.”) However, under
certain conditions, we will have
d
∗ = p
∗
,
so that we can solve the dual problem in lieu of the primal problem. Lets
see what these conditions are.
Suppose f and the gi
’s are convex,3 and the hi
’s are affine.4 Suppose
further that the constraints gi are (strictly) feasible; this means that there
exists some w so that gi(w) < 0 for all i.
3When f has a Hessian, then it is convex if and only if the hessian is positive semidefinite. For instance, f(w) = w
T w is convex; similarly, all linear (and affine) functions
are also convex. (A function f can also be convex without being differentiable, but we
won’t need those more general definitions of convexity here.)
4
I.e., there exists ai
, bi
, so that hi(w) = a
T
i w + bi
. “Affine” means the same thing as
linear, except that we also allow the extra intercept term bi
.
10

"""
 
result = chain.invoke({'text': text})

# print("Merged Notes and Quiz:\n", result)

print(chain.get_graph().draw_ascii())