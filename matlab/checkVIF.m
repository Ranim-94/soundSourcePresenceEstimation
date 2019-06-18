function [coll, VIF] = checkVIF(x)
% Checks for multicollinearity in supposedly independant variables
% x: Matrix (samples, variables)

n_var = size(x, 2);
VIF = zeros(n_var, 1);
for ind_var = 1:n_var
	mdl = fitlm(x(:, [1:ind_var-1 ind_var+1:end]), x(:, ind_var));
	VIF(ind_var) = 1/(1-mdl.Rsquared.Ordinary);
end
coll = VIF > 5;

end