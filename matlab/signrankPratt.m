function [p, h] = signrankPratt(x, y, method, zero_method)

d = round(x-y);

if strcmp(zero_method, 'wilcoxon')
    d = d(d~=0);
end

c = length(d);


[r, ~] = tiedrank(abs(d),0,0,zeros(c, 1));

if strcmp(method, 'normal')
    rp = sum(r.*(d>0));
    rm = sum(r.*(d<0));

    if strcmp(zero_method, 'split')
        % Separate 0 differences in two
        rz = sum(r.*(d==0));
        rp = rp + rz/2;
        rm = rm + rz/2;
    end

    T = min(rp, rm);

    mn = c*(c+1)/4;
    se = c*(c+1)*(2*c+1);

    if strcmp(zero_method, 'pratt')
        % Remove zeros before adjustement
        r = r(d~=0);
        % https://github.com/scipy/scipy/issues/6805
        % https://github.com/scipy/scipy/blob/v1.2.1/scipy/stats/morestats.py#L2709-L2806
        % http://faculty.washington.edu/fscholz/SpringStat425_2013/Stat425Ch3.pdf
        cz = sum(d==0);
        mn = mn - cz*(cz+1)/4;
        se = se - cz*(cz+1)*(2*cz+1);
    end



    % Find and count repetitions
    [repnum, ~] = hist(r, unique(r));
    repnum = repnum(repnum~=1);
    if ~isempty(repnum)
        % Adjustement value for ties
        tieadj = sum(repnum.*(repnum.*repnum-1))/2;
    end

    se = sqrt((se-tieadj)/24);
    z = (T-mn)/se;
    p = 2*normcdf(-abs(z),0,1);
    
elseif strcmp(method, 'sim')
    rd = r.*sign(d);
    rdr = abs(rd(d~=0));
    Vstar = max(sum(rd(rd>0)), -sum(rd(rd<0)));
    tic
    for iSim = 1:10000
        rBin = binornd(1, 0.5, sum(sign(d)~=0), 1)*2-1;
        Vvec(iSim) = sum(rdr(rBin>0));
    end
    toc
    p = mean(Vvec>=Vstar);
end

h = (p<=0.05);

end

