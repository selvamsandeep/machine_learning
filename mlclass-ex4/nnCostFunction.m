function [J grad ] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed 

% Part 2: Implement the backpropagation algorithm to compute the gradients
%        Theta1_grad and Theta2_grad. You should return the partial derivatives of
%        the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%        Theta2_grad, respectively. After implementing Part 2, you can check
%       that your implementation is correct by running checkNNGradients

% Note: The vector y passed into the function is a vector of labels
%      containing values from 1..K. You need to map this vector into a 
%   binary vector of 1's and 0's to be used with the neural network
%   cost function.

% Hint: We recommend implementing backpropagation using a for-loop
% over the training examples if you are implementing it for the 
% first time.

% Part 3: Implement regularization with the cost function and gradients.

%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X =[ones(m,1) X];

A2 = sigmoid(X*Theta1');
A2 =[ones(m,1) A2];
A3 = sigmoid(A2*Theta2');
H = A3;
%convertion y in to m X K  matix  for K classification
%we need to recode the labels as vectors containing only values 0 or  1, so that
%For example, if x(i) is an image of the digit 5, then the corresponding
%y(i) (that you should use with the cost function) should be a 10-dimensional
%vector with y5 = 1, and the other elements equal to 0.
yNew = zeros(m,num_labels);
for i =1:num_labels,
   yBinary = (y == i);
   yNew(:,i) = yBinary;
 
end   

for i = 1:m
   J =  J+(log(H(i,:))*yNew(i,:)' + log(1-H(i,:))*(1-yNew(i,:)'));
   
 end  
%J =  -(1/m)*sum( log(H')*yNew + log(1-H')*(1-yNew));
%J = -(1/m)* sum( log(H)*yNew' + log(1-H)*(1-yNew'));
%J =  (-1/m)*J ; cost function without requrlarization

%regularized cost 
temp_theta1 =Theta1(:,2:end);
temp_theta1 =temp_theta1'(:);

temp_theta2 =Theta2(:,2:end);
temp_theta2 =temp_theta2'(:);

J = (-1/m)*J + (lambda*(temp_theta1'*temp_theta1 + temp_theta2'*temp_theta2))/(2*m);
%back propagation

  delta3 = H -yNew;  
  delta2 = delta3*Theta2(:,2:end).* sigmoidGradient(X*Theta1');  
   
  Delta2 = Theta2_grad + delta3'*A2;
  Delta1  = Theta1_grad  + delta2'*X ;
  
  temp1 = Theta1;
  temp1(:,1) =0;
  temp2 = Theta2;
  temp2(:,1) =0;
  Theta2_grad = (1/m)*Delta2 + (lambda/m)*temp2;
  Theta1_grad = (1/m)*Delta1 + (lambda/m)*temp1;
  
  
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
