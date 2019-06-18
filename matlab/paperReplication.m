clearvars; close all; clc;

%% 2.1 Corpus generation
%% Corpus visualization (Figure 1)

visualizeCorpus();

%% 2.3 Corpus validation
%% Recorded-Replicated comparison (Table 1)

[res_mat, ~] = loadResponses();

iRec = [1 2 3 4 5 6];
iRep = [7 14 16 9 23 25];

res_rec = permute(res_mat(iRec, :, :), [3, 2, 1]);
res_rep = permute(res_mat(iRep, :, :), [3, 2, 1]);

pWP = zeros(length(iRec), size(res_rec, 2));
hWP = zeros(length(iRec), size(res_rec, 2));
for iSc = 1:length(iRec)
    for iQ = 1:size(res_rec, 2)
        % Table 1: hWP
        [pWP(iSc, iQ), hWP(iSc, iQ)] = signrankPratt(res_rec(:, iQ, iSc)', res_rep(:, iQ, iSc)', 'normal', 'pratt');
    end
end

%% PCA original and replicated scenes (Figure 2)

[res_mat, res_mean] = loadResponses();

fontSize = 14;

set(0, 'defaultTextFontSize',fontSize)

[coeff, score, ~, ~, explained, ~] = pca([res_mean(1:25, 1:5) 10-res_mean(1:25, 1:5)]);
figure(4), clf, biplot(coeff(:, 1:2), ...
    'VarLabels', {'Pleasant', 'Lively', 'Noisy', 'Interesting', 'Calm', 'Unpleasant', 'Inert', 'Quiet', 'Boring', 'Chaotic'}), hold on,
axis([-0.6 0.6 -0.6 0.6]);
axis square
norm_score = sqrt(max(sum(coeff(:, 1:2).^2,2)))/max(max(abs(score(:, 1:2))));
score = score*sqrt(max(sum(coeff(:, 1:2).^2,2)))/max(max(abs(score(:, 1:2))));

mu = repmat(mean([res_mean(1:25, 1:5) 10-res_mean(1:25, 1:5)]), 25, 1);

for iS = [1 7]% [(1:6) 7 14 16 9 23 25], pairs are [1 7], [2 14], [3 16], [4 9], [5 23], [6 25]
    score_ind = ([squeeze(res_mat(iS, 1:5, :))' 10-squeeze(res_mat(iS, 1:5, :))']-repmat(mu(1, :), 23, 1))/coeff';
    score_ind = score_ind*norm_score;
    score_ind = score_ind(:, 1:2);
    cov_score = cov(score_ind);
    [V,D] = eig(cov_score);
    [imaxD, ~] = find(D == max(max(D)));
    maxV = V(:, imaxD);
    angS = atan2(maxV(2), maxV(1));
    ellipse(sqrt(D(2, 2)), sqrt(D(1, 1)), angS, score(iS, 1), score(iS, 2));
end

plot([score((1:6), 1)'; score([7 14 16 9 23 25], 1)'], [score((1:6), 2)'; score([7 14 16 9 23 25], 2)'], '-k')
quiver(score((1:6), 1)', score((1:6), 2)', score([7 14 16 9 23 25], 1)'-score((1:6), 1)', score([7 14 16 9 23 25], 2)'-score((1:6), 2)',0, 'k')
text((score(7, 1)'+score(1, 1)')/2+0.01,(score(7, 2)'+score(1, 2)')/2-0.03,'P1')
text((score(14, 1)'+score(2, 1)')/2+0.02,(score(14, 2)'+score(2, 2)')/2,'P3')
text((score(16, 1)'+score(3, 1)')/2-0.1,(score(16, 2)'+score(3, 2)')/2-0.05,'P4')
text((score(9, 1)'+score(4, 1)')/2-0.02,(score(9, 2)'+score(4, 2)')/2-0.05,'P8')
text((score(23, 1)'+score(5, 1)')/2+0.02,(score(23, 2)'+score(5, 2)')/2,'P15')
text((score(25, 1)'+score(6, 1)')/2-0.07,(score(25, 2)'+score(6, 2)')/2+0.04,'P18')
xlabel(['Component 1 (' num2str(round(100*explained(1))/100) '\% explained)'], 'FontSize',fontSize)
ylabel(['Component 2 (' num2str(round(100*explained(2))/100) '\% explained)'], 'FontSize',fontSize)
set(gca, 'FontSize', fontSize)

disp(['The first 2 dimensions explain resp. ' num2str(explained(1)) '% and ' num2str(explained(2)) '% of the variance. (' num2str(explained(1)+explained(2)) '% total).']);
% export_fig pca_p1.eps -eps -transparent

%% PCA simulated scenes (Figure 3)

fontSize = 14;

set(0, 'defaultTextFontSize',fontSize)
[coeff,score,latent,tsquared,explained,mu] = pca([res_mean(26:end, 1:5) 10-res_mean(26:end, 1:5)]);
figure(5), clf, biplot(coeff(:, 1:2), 'Scores', score(:, 1:2), ...
    'VarLabels', {'Pleasant', 'Lively', 'Noisy', 'Interesting', 'Calm', 'Unpleasant', 'Inert', 'Quiet', 'Boring', 'Chaotic'}, 'MarkerSize', 10), hold on,
score1 = [res_mean(1:25, 1:5) 10-res_mean(1:25, 1:5)]*coeff;
score1 = score1*sqrt(max(sum(coeff(:, 1:2).^2,2)))/max(max(abs(score1(:, 1:2))));
plot(score1(:, 1), score1(:, 2), 'x')
axis([-0.6 0.6 -0.6 0.6]);
axis square

xlabel(['Component 1 (' num2str(round(100*explained(1))/100) '\% explained)'], 'FontSize',fontSize)
ylabel(['Component 2 (' num2str(round(100*explained(2))/100) '\% explained)'], 'FontSize',fontSize)
set(gca, 'FontSize', fontSize)

disp(['The first 2 dimensions explain resp. ' num2str(explained(1)) '% and ' num2str(explained(2)) '% of the variance. (' num2str(explained(1)+explained(2)) '% total).']);
% export_fig pca_sim.eps -eps -transparent

%% 3.1 Acoustical indicators for soundscape description

% Global indicators
globalIndicators;
% Find best alpha and beta values
activityProfiles;
searchThreshold;
% Source-specific indicators
sourceIndicators;
% Rearrange indicators
orderPhys;

%% 4.1 Perceptual pleasantness model
%% Perceptual correlations (Table 2)

[~, res_mean] = loadResponses();

[rPercPerc, pPercPerc] = pearsonCoefs(res_mean, res_mean);
% Table 2: rPercPerc

%% Perceptual models (Eqn. 5-6, Table 3)

[~, res_mean] = loadResponses();
predNames = {'P', 'L', 'OL', 'I', 'C', 'LT', 'TT', 'TV', 'TB'};
P = res_mean(:, 1);

% Pleasantness perceptual models n = 100
percMdls = mdlPred(P, res_mean, predNames, [3 6 7 8 9]);

%% 4.2 Physical pleasantness models
%% Physical correlations (Table 4)

[~, res_mean] = loadResponses();
res_mean = res_mean(7:end, :);

load phys_ordered;

diff_Leq = mean(phys.playbackLeq-phys.Leq);

% Build physical indicators matrix
scene_names = phys.name;
phys_inds = {'LAeq', 'LA50', 'Leq', 'L10', 'L50', 'L90', 'L50_1k', 'L10-L90', 'TFSD_500', 'TFSD_4k', ...
    'Leq_s_t', 'Leq_s_v', 'Leq_s_b', 'em_t', 'em_v', 'em_b', 't_pres_tf_t', 't_pres_tf_v', 't_pres_tf_b'};
phys_mat = [phys.LAeq, phys.LA50, phys.Leq, phys.L10, phys.L50, phys.L90, phys.L50_1k, phys.L10-phys.L90, phys.TFSD_500, phys.TFSD_4k, ...
    phys.Leq_s+diff_Leq, phys.emergence, phys.t_pres_tf];

phys_mat(isinf(phys_mat)&(phys_mat<0)) = 0;

% Find scenes where only one source is present
i_inf = isnan(mean(phys_mat, 2))|isinf(mean(phys_mat, 2));

scene_names = scene_names(~i_inf, :);
phys_mat = phys_mat(~i_inf, :);
res_mean = res_mean(~i_inf, :);

[rPercPhy, pPercPhy] = pearsonCoefs(phys_mat, res_mean);
% Table 4: rPercPhy

%% Physical models (Eqn. 7-9, Table 5, Table 7 right)

[~, res_mean] = loadResponses();
P = res_mean(7:end, 1);

load phys_ordered;

phys.emergence(isinf(phys.emergence)) = 0;
phys_mat = [phys.L50+103 phys.t_pres_tf];
predNames = {'L50', 'T_alpha_beta_T', 'T_alpha_beta_V', 'T_alpha_beta_B'};
mdl_p = mdlPred(P, phys_mat, predNames);

k = size(mdl_p(1).mdl.Coefficients, 1)-1;
P_est = mdl_p(1).mdl.Fitted.Response;
P_diff = P-P_est;

rmse_phys1 = sqrt(mean(P_diff.^2));
[r, p] = corrcoef(P, P_est);
r_phys1 = r(1, 2);
p_phys1 = p(1, 2);

rmse_phys1_rep = sqrt(mean(P_diff(1:19).^2));
[r, p] = corrcoef(P(1:19, 1), P_est(1:19));
r_phys1_rep = r(1, 2);
p_phys1_rep = p(1, 2);

rmse_phys1_sim = sqrt(mean(P_diff(20:end).^2));
[r, p] = corrcoef(P(20:end, 1), P_est(20:end));
r_phys1_sim = r(1, 2);
p_phys1_sim = p(1, 2);

n = 19;
R2_phys1_rep = 1-(var(P_diff(1:19))/var(P(1:19, 1)));
R2adj_phys1_rep = 1-(1-R2_phys1_rep)*((n-1)/(n-(k+1)));
n = 75;
R2_phys1_sim = 1-(var(P_diff(20:end))/var(P(20:end, 1)));
R2adj_phys1_sim = 1-(1-R2_phys1_sim)*((n-1)/(n-(k+1)));


% Ricciardi et al. (2014)
% Original model: P_est = 18.08 - 0.19*phys_mat(:, 1) - 0.06*phys_mat(:, 5)+0.0969;
phys_mat = [phys.L50+103 phys.L10-phys.L90];
mdl_r14 = fitglm(phys_mat, P);
P_est = mdl_r14.Fitted.Response;

[r, ~] = corrcoef(P, P_est);
r_r14 = r(1, 2);
rmse_r14 = sqrt(mean((P_est-P).^2));

% Aumond et al. (2017)
% Original model: P = 15.48 - 0.25*L50 + 15.82*phys_mat(:, 7) + 16.82*phys_mat(:, 8)+1.070;
phys_mat = [phys.L50_1k+103 phys.TFSD_500 phys.TFSD_4k];
phys_mat(:, 1) = phys_mat(:, 1)+103;

mdl_a17 = fitglm(phys_mat, P);
P_est = mdl_a17.Fitted.Response;

[r, ~] = corrcoef(P, P_est);
r_a17 = r(1, 2);
rmse_a17 = sqrt(mean((P_est-P).^2));

%% 4.3 Time of presence prediction through deep learning
%% Deep learning predictions (Table 7-left)

[~, res_mean] = loadResponses();

t_pred_real = csvread('test_recrep_pred.txt');
t_pred_rec = t_pred_real(1:6, :);
t_pred_rep = t_pred_real(7:end, :);
t_pred_sim = csvread('test_sim_pred.txt');
load phys_ordered;
L50_rec = phys.L50([7 14 16 9 23 25]-6)+103;
L50_rep = phys.L50(1:19)+103;
L50_sim = phys.L50(20:end)+103;

p_pred_rec = 16.74 - 0.18*L50_rec + 1.01*t_pred_rec(:, 3);
p_pred_rep = 16.74 - 0.18*L50_rep + 1.01*t_pred_rep(:, 3);
p_pred_sim = 16.74 - 0.18*L50_sim + 1.01*t_pred_sim(:, 3);

% figure(1), clf, plot([res_mean(1:6, 1) p_pred_rec]), grid on;
% figure(2), clf, plot([res_mean(7:25, 1) p_pred_rep]), grid on;
% figure(3), clf, plot([res_mean(26:end, 1) p_pred_sim]), grid on;

rmse_rec = sqrt(mean((res_mean(1:6, 1)-p_pred_rec).^2));
rmse_rep = sqrt(mean((res_mean(7:25, 1)-p_pred_rep).^2));
rmse_sim = sqrt(mean((res_mean(26:end, 1)-p_pred_sim).^2));
rmse_all = sqrt(mean((res_mean(:, 1)-[p_pred_rec; p_pred_rep; p_pred_sim]).^2));

[r, p] = corrcoef(res_mean(1:6, 1), p_pred_rec);
r_rec = r(1, 2); p_rec = p(1, 2);
[r, p] = corrcoef(res_mean(7:25, 1), p_pred_rep);
r_rep = r(1, 2); p_rep = p(1, 2);
[r, p] = corrcoef(res_mean(26:end, 1), p_pred_sim);
r_sim = r(1, 2); p_sim = p(1, 2);
[r, p] = corrcoef(res_mean(:, 1), [p_pred_rec; p_pred_rep; p_pred_sim]);
r_all = r(1, 2); p_all = p(1, 2);

k = 2;
n = 6;
R2_rec = 1-(var((p_pred_rec-res_mean(1:6, 1)))/var(res_mean(1:6, 1)));
R2adj_rec = 1-(1-R2_rec)*((n-1)/(n-(k+1)));
n = 19;
R2_rep = 1-(var((p_pred_rep-res_mean(7:25, 1)))/var(res_mean(7:25, 1)));
R2adj_rep = 1-(1-R2_rep)*((n-1)/(n-(k+1)));
n = 75;
R2_sim = 1-(var((p_pred_sim-res_mean(26:end, 1)))/var(res_mean(26:end, 1)));
R2adj_sim = 1-(1-R2_sim)*((n-1)/(n-(k+1)));
n = 100;
R2_all = 1-(var(([p_pred_rec; p_pred_rep; p_pred_sim]-res_mean(:, 1)))/var(res_mean(:, 1)));
R2adj_all = 1-(1-R2_all)*((n-1)/(n-(k+1)));

