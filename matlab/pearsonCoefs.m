function [cCoef, cSign] = pearsonCoefs(x1, x2)
% Returns Pearson correlation coefficients cCoef and their significance cSign (0: p>0.05, 1: 0.01<p<=0.05, 2: p<=0.01)

if isequal(x1, x2)
    nVar = size(x1, 2);
    cCoef = eye(nVar);
    pVal = zeros(nVar);
    for iV1 = 1:nVar
        for iV2 = iV1+1:nVar
            [r, p] = corrcoef(x1(:, iV1), x1(:, iV2));
            cCoef(iV1, iV2) = r(1, 2);
            pVal(iV1, iV2) = p(1, 2);
        end
    end
else
    nVar1 = size(x1, 2);
    nVar2 = size(x2, 2);
    cCoef = zeros(nVar1, nVar2);
    pVal = zeros(nVar1, nVar2);
    for iV1 = 1:nVar1
        for iV2 = 1:nVar2
            [r, p] = corrcoef(x1(:, iV1), x2(:, iV2));
            cCoef(iV1, iV2) = r(1, 2);
            pVal(iV1, iV2) = p(1, 2);
        end
    end
end

cSign = zeros(size(pVal));
cSign(pVal<=0.01) = 2;
cSign(pVal<=0.05 & pVal>0.01) = 1;
cSign(pVal>0.05 | pVal==0) = 0;

end