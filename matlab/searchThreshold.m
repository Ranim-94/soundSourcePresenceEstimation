clear all, close all, clc;

%% Critical bands-based time of presence
[~, res_mean] = loadResponses();
res_mean = res_mean(7:end, 7:end);

load act_profiles; % Generated by activityProfiles.m

start_a = 1;
end_a = size(res_mean, 1);

alpha_t = (-20:10);
beta_t = (-20:10);

[spl, freq] = iso226(0);

rAB = zeros(length(alpha_t), length(beta_t));
for iA = 1:length(alpha_t)
    for iB = 1:length(beta_t)
        t_pres_tf = zeros(size(res_mean));
        for iSrc = 1:size(res_mean, 2)
            for i_a = 1:size(res_mean, 1)
                t_pres_tf(i_a, iSrc) = timePresence(L_tob_s{i_a, iSrc}, L_tob_o{i_a, iSrc}, alpha_t(iA), beta_t(iB), spl', 0);
            end
            [r, ~] = corrcoef(res_mean(:, iSrc), t_pres_tf(:, iSrc));
            rAB(iA, iB, iSrc) = round(r(1, 2)*100)/100;
%             disp(['(alpha, beta) = (' num2str(alpha_t(iA)) ', ' num2str(beta_t(iB)) '), r = ' num2str(mean(rAB(iA, iB, :), 3))])
        end
    end
end
figure(1), clf, mesh(beta_t, alpha_t, mean(rAB, 3))

[max_alpha, iAOpt] = max(mean(rAB, 3));
[max_corr_tf, iBOpt] = max(max_alpha);
iAOpt = iAOpt(iBOpt);
alpha_opt = alpha_t(iAOpt); beta_opt = beta_t(iBOpt);

r_s = squeeze(rAB(iAOpt, iBOpt, :));

disp(['(alpha_opt, beta_opt) = (' num2str(alpha_opt) ', ' num2str(beta_opt) ') : r = (' num2str(r_s(1)) ', ' num2str(r_s(2)) ', ' num2str(r_s(3)) ').'])
