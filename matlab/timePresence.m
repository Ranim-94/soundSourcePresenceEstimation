function [t_pres_tf, pres_tf] = timePresence(L_tob_s, L_tob_o, alpha_t, beta_t, spl, L_diff)

if isinf(L_tob_s)
    t_pres_tf = 0;
    pres_tf = zeros(1, size(L_tob_s, 2));
else
    assert(all(size(L_tob_s)==size(L_tob_o)))
    
    % Apply hearing threshold curve
    L_tob_s(L_tob_s+L_diff < repmat(spl, 1, size(L_tob_s, 2))) = -Inf;
    
    diff_tf = L_tob_s - L_tob_o;
    diff_tf(isnan(diff_tf)) = -Inf;
    
    i_diff_tf = diff_tf > alpha_t;
    
    diff_t = nansum(i_diff_tf.*diff_tf)./max(sum(i_diff_tf), ones(1, size(i_diff_tf, 2)));
    diff_t(diff_t == 0) = -Inf;
    
    t_pres_tf = mean(diff_t > beta_t);
    pres_tf = diff_t > beta_t;
end

end