function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
num=size(data,2);  
x=data;  
a1=x;  
%根据sparse autoencider理论求出隐藏层和最后一层的结果
z2=W1*a1+repmat(b1,1,num);  
a2=sigmoid(z2);  
z3=W2*a2+repmat(b2,1,num);  
a3=sigmoid(z3);

h=a3;  
y=x;
%第一项
squared_error=0.5*sum((h-y).^2,1);  
rho=1/num*sum(a2,2);  
%第二项 正则项==>防止过拟合   第三项 KL散度 ==> 保证稀疏度
sparsity_penalty= beta*sum(sparsityParam.*log(sparsityParam./rho)+(1-sparsityParam).*log((1-sparsityParam)./(1-rho)));  
%总的cost function
cost=1/num*sum(squared_error)+lambda/2*(sum(sum(W1.^2))+sum(sum(W2.^2))) + sparsity_penalty;  
%根据总的表达式，进行参数的优化
grad_z3=a3.*(1-a3);  
delta_3=-(y-a3).*grad_z3;  
grad_z2=a2.*(1-a2);  
delta_2=(W2'*delta_3+repmat(beta*(-sparsityParam./rho+(1-sparsityParam)./(1-rho)),1,num)).*grad_z2;  

Delta_W2=delta_3*a2';  
Delta_b2=sum(delta_3,2);  
Delta_W1=delta_2*a1';  
Delta_b1=sum(delta_2,2);  

W1grad=1/num*Delta_W1+lambda*W1;  
W2grad=1/num*Delta_W2+lambda*W2;  
b1grad=1/num*Delta_b1;  
b2grad=1/num*Delta_b2;  

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

