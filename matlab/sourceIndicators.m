clear all, close all, clc;

dataset = {'rep', 'sim'};
dataset_path = {'audio/rep/', 'audio/sim/'};

sources = {'T', 'V', 'B'};

sr = 32e3;
l_frame = 32768;

alpha_opt = -14;
beta_opt = -7;
[spl, freq] = iso226(0);
spl = spl';

for i_dataset = 2:length(dataset)
    disp(['----- DATASET ' dataset{i_dataset} ' -----']);
    load([dataset{i_dataset} '_exp.mat']);
    
    if i_dataset == 1
        source_names{1} = {'traffic', 'cityCar', 'roadCar'};
        source_names{2} = {'crowd', 'voice', 'schoolyard'};
        source_names{3} = {'bird', 'park'};
    else
        source_names{1} = {'traffic', 'cityCar', 'roadCar'};
        source_names{2} = {'crowd', 'voice'};
        source_names{3} = {'bird'};
    end
    
    a_path = exp_scenes.name;
    for i_a = 1:length(a_path)
        a_path_temp = [a_path{i_a} '.wav'];
        disp(a_path_temp);
        
        %% Info
        src.name{i_a, 1} = a_path_temp(1:end-4);
        src.type{i_a, 1} = a_path_temp(1:end-6);
        
        %% Audio pre-pocessing
        [x, sr_x] = audioread([dataset_path{i_dataset} a_path_temp]);
        x = resample(x, sr, sr_x);
        
        %% Leq global (for hearing threshold)
        scf = itaAudio(x, sr, 'time');
        SC = ita_spk2level(scf, 0, 'added', 0);
        L_global = 20*log10(SC.freq);
        L_diff = exp_scenes.Leq(i_a)-L_global;
        
        %% Channels
        c_path = dir([dataset_path{i_dataset} src.name{i_a} '*']);
        c_path = c_path(2:end); % Remove mix
        c_path = {c_path.name};
        
        for i_s = 1:length(source_names) % Source (T, V, B)
            disp([' -> ' sources{i_s}])
            i_path_s = zeros(size(c_path));
            for i_ss = 1:length(source_names{i_s}) % Subsource (traffic, cityCar, ...)
                % Indices of channels associated to current sources
                i_path_s = i_path_s | not(cellfun('isempty', strfind(c_path, source_names{i_s}{i_ss})));
            end
            c_path_s = c_path(i_path_s); % List of channels associated to current sources
            c_path_o = c_path(~i_path_s); % List of channels for other sources
            if ~isempty(c_path_s)
                %% Current source
                x = 0;
                for i_c = 1:length(c_path_s)
                    disp(['    S: ' c_path_s{i_c}]);
                    [x_c, sr_c] = audioread([dataset_path{i_dataset} c_path_s{i_c}]);
                    x_c = resample(x_c, sr, sr_c);
                    x = x + x_c;
                end
                
                % Sound level
                n_x = floor(length(x)/l_frame);
                L_tob_n = zeros(29, n_x);
                for indf = 1:n_x
                    if ~isempty(find(x((indf-1)*l_frame+1:indf*l_frame), 1))
                        xf = itaAudio(x((indf-1)*l_frame+1:indf*l_frame), sr, 'time');
                        X = ita_spk2frequencybands(xf,'mode','filter');
                        L_tob_n(:, indf) = X.freq(:, 1);
                    else
                        L_tob_n(:, indf) = 0;
                    end
                end
                % Leq
                L_tob_s = 20*log10(L_tob_n);
                L_s = 20*log10(sqrt(sum(L_tob_n.^2)));
                
                
                %% Other sources
                x_o = zeros(size(x));
                for i_c = 1:length(c_path_o)
                    disp(['    O: ' c_path_o{i_c}]);
                    [x_c, sr_c] = audioread([dataset_path{i_dataset} c_path_o{i_c}]);
                    x_c = resample(x_c, sr, sr_c);
                    x_o = x_o + x_c;
                end
                
                % Sound level
                n_x = floor(length(x_o)/l_frame);
                L_tob_n = zeros(29, n_x);
                for indf = 1:n_x
                    if ~isempty(find(x_o((indf-1)*l_frame+1:indf*l_frame), 1))
                        xf = itaAudio(x_o((indf-1)*l_frame+1:indf*l_frame), sr, 'time');
                        X = ita_spk2frequencybands(xf,'mode','filter');
                        L_tob_n(:, indf) = X.freq(:, 1);
                    else
                        L_tob_n(:, indf) = 0;
                    end
                end
                % Leq
                L_tob_o = 20*log10(L_tob_n);
                L_o = 20*log10(sqrt(sum(L_tob_n.^2)));
                
                % Equivalent sound levels
                src.Leq_s(i_a, i_s) = 10*log10(mean(10.^(L_s/10)));
                src.Leq_o(i_a, i_s) = 10*log10(mean(10.^(L_o/10)));
                src.emergence(i_a, i_s) = src.Leq_s(i_a, i_s) - src.Leq_o(i_a, i_s);
                %% Time of presence
                src.t_pres_tf(i_a, i_s) = timePresence(L_tob_s, L_tob_o, alpha_opt, beta_opt, spl, L_diff);
            else
                src.t_pres_tf(i_a, i_s) = 0;
                src.emergence(i_a, i_s) = -Inf;
                src.Leq_s(i_a, i_s) = -Inf;
                src.Leq_o(i_a, i_s) = NaN;
            end
            disp([' ---> Estimated time of presence T(alpha, beta) = ' num2str(src.t_pres_tf(i_a, i_s)) '.'])
            
        end
    end
    
    %% Save
    save([dataset{i_dataset} '_src.mat'], 'src');
end

