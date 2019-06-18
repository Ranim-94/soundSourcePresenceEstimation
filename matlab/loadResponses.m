function [res_mat, res_mean] = loadResponses()

res_path = 'assessments/';
res_path_f = dir(res_path);
res_path_f = res_path_f(3:end);

opt = {'Delimiter','\t', 'MultipleDelimsAsOne',true};

res_mat = zeros(100, 9, 1);
for i_res_path = 1:length(res_path_f)
    fid = fopen([res_path res_path_f(i_res_path).name '/Pt_.txt'],'r');
    C = textscan(fid,'%f%f%f%f%f%f%f%f%f%f',opt{:});
    fclose(fid);
    M = cell2mat(C);
    for i_sc = 1:size(M, 1)
        res_mat(M(i_sc, 1), :, i_res_path) = M(i_sc, 2:end);
    end
    for i_sc = 1:100
        if isempty(find(M(:, 1)==i_sc, 1))
            res_mat(i_sc, :, i_res_path) = NaN*ones(1, 9, 1);
        end
    end
    
end

res_mean = nanmean(res_mat, 3);

end