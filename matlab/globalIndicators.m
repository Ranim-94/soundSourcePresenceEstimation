clear all, close all, clc;

dataset = {'rep', 'sim'};
dataset_path = {'audio/rep/', 'audio/sim/'};

sr = 32e3;
l_frame = 32768;
load phys_ordered;
for i_dataset = 2:length(dataset)
    disp(['----- DATASET ' dataset{i_dataset} ' -----']);
    load([dataset{i_dataset} '_exp.mat']);
    a_path = exp_scenes.name;
    for i_a = 1:length(a_path)
        a_path_temp = [a_path{i_a} '.wav'];
        disp(a_path_temp);
        
        %% Info
        glb.name{i_a, 1} = a_path_temp(1:end-4);
        glb.type{i_a, 1} = a_path_temp(1:end-6);
        
        %% Audio pre-processing
        [x, sr_x] = audioread([dataset_path{i_dataset} a_path_temp]);
        x = resample(x, sr, sr_x);
        
        %% Playback Leq
        glb.playbackLeq(i_a, 1) = exp_scenes.Leq(i_a);
        
        %% Levels
        %% LA50
        n_x = floor(length(x)/l_frame);
        LA_tob_t = zeros(29, n_x);
        for indf = 1:n_x
            if ~isempty(find(x((indf-1)*l_frame+1:indf*l_frame), 1))
                xf = itaAudio(x((indf-1)*l_frame+1:indf*l_frame), sr, 'time');
                X = ita_spk2frequencybands(xf, 'mode', 'filter', 'weighting', 'A');
                LA_tob_t(:, indf) = X.freq(:, 1);
            else
                LA_tob_t(:, indf) = 0;
            end
        end
        LA_t = 20*log10(sqrt(sum(LA_tob_t.^2, 1)));
        glb.LAeq(i_a, 1) = 10*log10(mean(10.^(LA_t/10)));
        if ~isempty(find(~isinf(LA_t), 1))
            LA_t_s = sort(LA_t);
            LA_t_s = LA_t_s(~isinf(LA_t_s));
            % LA50
            glb.LA50(i_a, 1) = LA_t_s(floor(0.5*length(LA_t_s)));
        else
            glb.LA50(i_a, 1) = -Inf;
        end
        
        % L10, L50, L90
        n_x = floor(length(x)/l_frame);
        L_tob_t = zeros(29, n_x);
        for indf = 1:n_x
            if ~isempty(find(x((indf-1)*l_frame+1:indf*l_frame), 1))
                xf = itaAudio(x((indf-1)*l_frame+1:indf*l_frame), sr, 'time');
                X = ita_spk2frequencybands(xf, 'mode', 'filter');
                L_tob_t(:, indf) = X.freq(:, 1);
            else
                L_tob_t(:, indf) = 0;
            end
        end
        L_t = 20*log10(sqrt(sum(L_tob_t.^2, 1)));
        glb.Leq(i_a, 1) = 10*log10(mean(10.^(L_t/10)));
        if ~isempty(find(~isinf(L_t), 1))
            L_t_s = sort(L_t);
            L_t_s = L_t_s(~isinf(L_t_s));
            % L10
            glb.L10(i_a, 1) = L_t_s(max(floor(0.9*length(L_t_s)), 1));
            % L50
            glb.L50(i_a, 1) = L_t_s(max(floor(0.5*length(L_t_s)), 1));
            % L90
            glb.L90(i_a, 1) = L_t_s(max(floor(0.1*length(L_t_s)), 1));
        else
            glb.L10(i_a, 1) = -Inf;
            glb.L50(i_a, 1) = -Inf;
            glb.L90(i_a, 1) = -Inf;
        end
        
        % L50_1k
        L_1k_t = 20*log10(sqrt(L_tob_t(18, :).^2));
        if ~isempty(find(~isinf(L_1k_t), 1))
            L_1k_t_s = sort(L_1k_t);
            L_1k_t_s = L_1k_t_s(~isinf(L_1k_t_s));
            % LA50
            glb.L50_1k(i_a, 1) = L_1k_t_s(floor(0.5*length(L_1k_t_s)));
        else
            glb.L50_1k(i_a, 1) = -Inf;
        end
        
        % TFSD
        glb.TFSD_500(i_a, 1) = tfsd(x, sr, l_frame, 15); 
        glb.TFSD_4k(i_a, 1) = tfsd(x, sr, l_frame/8, 24);
        
    end
    
    %% Save
    save([dataset{i_dataset} '_glb.mat'], 'glb');
end

