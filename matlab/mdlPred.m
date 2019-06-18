function [mdls] = mdlPred(y, X, predNames, iPred, nBest)
% y: data to predict, (n_obs*1) vector
% X: predictors, (n_obs*n_pred) matrix
% predNames: Cell array of predictor names
% iPred: indices of predictors to use (default: all)
% nBest: number of best models to return (default: 5)

if nargin<5; nBest = 5; end;
if (nargin<4 || isempty(iPred)); iPred = (1:size(X, 2)); end;

%% List predictors combinations where VIF < 5
predGroups = cell(0, 1);
for nPreds = 1:length(iPred)
    combPreds = combnk(iPred, nPreds);
    for iC = 1:size(combPreds, 1)
        if isempty(find(checkVIF(X(:, combPreds(iC, :))), 1))
            predGroups{end+1, 1} = combPreds(iC, :);
        end
    end
end

disp(['Found ' num2str(length(predGroups)) ' valid predictors combinations.'])

iMdl = 0;
for iC = 1:length(predGroups)
    mdl = fitglm(X(:, predGroups{iC}), y);
    if all(mdl.Coefficients.pValue<0.05) % All estimates are significant
        iMdl = iMdl+1;
        mdls(iMdl).mdl = mdl;
        mdls(iMdl).preds = {predNames{predGroups{iC}}};
        mdls(iMdl).R2_adj = mdl.Rsquared.Adjusted;
        [r, p] = corrcoef(mdl.Fitted.response, y);
        mdls(iMdl).r = [r(1, 2), p(1, 2)];
        mdls(iMdl).RMSE = sqrt(mean((mdl.Fitted.response-y).^2));
    end
end
[~, iBest] = sort([mdls.R2_adj], 'descend');
iBest = iBest(1:min(nBest, length(iBest)));
mdls = mdls(iBest);

end