function TFSD = tfsd(x, sr, l_frame, oct_band)

n_x = floor(length(x)/l_frame);
Leq_tob_n = zeros(29, n_x);
for indf = 1:n_x
    if ~isempty(find(x((indf-1)*l_frame+1:indf*l_frame), 1))
        xf = itaAudio(x((indf-1)*l_frame+1:indf*l_frame), sr, 'time');
        X=ita_spk2frequencybands(xf,'mode','filter');
        Leq_tob_n(:, indf)=X.freq(:, 1);
    else
        Leq_tob_n(:, indf) = 0;
    end
end
diff_Leq = abs(diff(diff(Leq_tob_n, 1, 1), 1, 2));
TFSD = mean(diff_Leq(oct_band,:)./sum(diff_Leq));

end