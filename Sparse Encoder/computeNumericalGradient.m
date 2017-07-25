function numgrad = computeNumericalGradient(J, theta)
% theta: a vector of parameters
% J: a function that outputs a real-number. Calling y = J(theta) will return the
% function value at theta. 
  
% Initialize numgrad with zeros
numgrad = zeros(size(theta));

%% ---------- YOUR CODE HERE --------------------------------------
% Instructions: 
% Implement numerical gradient checking, and return the result in numgrad.  
% You should write code so that numgrad(i) is (the numerical approximation to) the 
% partial derivative of J with respect to the i-th input argument, evaluated at theta.  
% I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
% respect to theta(i).
%                
% Hint: You will probably want to compute the elements of numgrad one at a time.
% J(theta)
epsilon = 1e-4;
for i = 1:size(numgrad)
%每次使用一个局部变量对
%theta的第i位进行+ - epsilon
%从而求得导数
    temp1 = theta;
    temp2 = theta;
    temp1(i) = theta(i) - epsilon; 
    temp2(i) = theta(i) + epsilon; 
    numgrad(i) = (J(temp2)-J(temp1))/(2*epsilon);
end
%% ---------------------------------------------------------------

