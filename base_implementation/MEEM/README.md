## MEEM: Robust Tracking via Multiple Experts using Entropy Minimization
### Folder structure
```
📦meem
 ┣ 📜calcIIF.cpp
 ┣ 📜calcIIF.m
 ┣ 📜createSampler.m
 ┣ 📜createSvmTracker.m
 ┣ 📜expertsDo.m
 ┣ 📜getFeatureRep.m
 ┣ 📜getIOU.m
 ┣ 📜getLogLikelihoodEntropy.m
 ┣ 📜im2colstep.c
 ┣ 📜im2colstep.m
 ┣ 📜initSampler.m
 ┣ 📜initSvmTracker.m
 ┣ 📜README.md
 ┣ 📜resample.m
 ┣ 📜RGB2Lab.m
 ┣ 📜rsz_rt.m
 ┣ 📜tracker_meem_initialize.m
 ┣ 📜tracker_meem_update.m
 ┣ 📜updateSample.m
 ┣ 📜updateSvmTracker.m
 ┗ 📜updateTrackerExperts.m
 ```


 ### [createSampler.m](meem/CreateSampler.m)
- Calls the fuction `createSampler()`
```matlab
sampler = [];
sampler.radius=1;
% sampler.scale_step = 1.1;
```

### [createSvmTracker.m](meem/CreateSvmTracker.m)
- Calls the fuction `createSvmTracker()`

```matlab
tracker = [];
tracker.sv_size = 500;% maxial 100 cvs
tracker.C = 100;
tracker.B = 80;% for tvm
tracker.B_p = 10;% for positive sv
tracker.lambda = 1;% for whitening
tracker.m1 = 1;% for tvm
tracker.m2 = 2;% for tvm
tracker.w = [];
tracker.w_smooth_rate = 0.0;
tracker.confidence = 1;
tracker.state = 0;
tracker.temp_count = 0;
tracker.output_feat_record = [];
tracker.feat_cache = [];
% tracker.experts = {};
tracker.confidence_exp = 1;
tracker.confidence = 1;
tracker.best_expert_idx = 1;
tracker.failure = false;
tracker.update_count = 0;
```
### [updateSvmTracker.m](meem/updateSvmTracker.m)
function to update the tracker
```matlab
function state = updateSvmTracker(state, sample, label, fuzzy_weight)

sample = [state.svm_tracker.pos_sv;state.svm_tracker.neg_sv; sample];
label = [ones(size(state.svm_tracker.pos_sv,1),1);zeros(size(state.svm_tracker.neg_sv,1),1);label];% positive:1 negative:0
sample_w = [state.svm_tracker.pos_w;state.svm_tracker.neg_w;fuzzy_weight];
       
pos_mask = label>0.5;
neg_mask = ~pos_mask;
s1 = sum(sample_w(pos_mask));
s2 = sum(sample_w(neg_mask));
        
sample_w(pos_mask) = sample_w(pos_mask)*s2;
sample_w(neg_mask) = sample_w(neg_mask)*s1;
        
C = max(state.svm_tracker.C*sample_w/sum(sample_w),0.001);


    state.svm_tracker.clsf = svmtrain( sample, label, ...%'kernel_function',@kfun,'kfunargs',{state.svm_tracker.struct_mat},...
       'boxconstraint',C,'autoscale','false','options',statset('MaxIter',5000));

%**************************
state.svm_tracker.w = state.svm_tracker.clsf.Alpha'*state.svm_tracker.clsf.SupportVectors;
state.svm_tracker.Bias = state.svm_tracker.clsf.Bias;
state.svm_tracker.clsf.w = state.svm_tracker.w;
% get the idx of new svs
sv_idx = state.svm_tracker.clsf.SupportVectorIndices;
sv_old_sz = size(state.svm_tracker.pos_sv,1)+size(state.svm_tracker.neg_sv,1);
sv_new_idx = sv_idx(sv_idx>sv_old_sz);
sv_new = sample(sv_new_idx,:);
sv_new_label = label(sv_new_idx,:);
        
num_sv_pos_new = sum(sv_new_label);
        
% update pos_dis, pos_w and pos_sv
pos_sv_new = sv_new(sv_new_label>0.5,:);
if ~isempty(pos_sv_new)
    if size(pos_sv_new,1)>1
        pos_dis_new = squareform(pdist(pos_sv_new));
    else
        pos_dis_new = 0;
    end
    pos_dis_cro = pdist2(state.svm_tracker.pos_sv,pos_sv_new);
    state.svm_tracker.pos_dis = [state.svm_tracker.pos_dis, pos_dis_cro; pos_dis_cro', pos_dis_new];
    state.svm_tracker.pos_sv = [state.svm_tracker.pos_sv;pos_sv_new];
    state.svm_tracker.pos_w = [state.svm_tracker.pos_w;ones(num_sv_pos_new,1)];
end
        
% update neg_dis, neg_w and neg_sv
neg_sv_new = sv_new(sv_new_label<0.5,:);
if ~isempty(neg_sv_new)
    if size(neg_sv_new,1)>1
        neg_dis_new = squareform(pdist(neg_sv_new));
    else
        neg_dis_new = 0;
    end
    neg_dis_cro = pdist2(state.svm_tracker.neg_sv,neg_sv_new);
    state.svm_tracker.neg_dis = [state.svm_tracker.neg_dis, neg_dis_cro; neg_dis_cro', neg_dis_new];
    state.svm_tracker.neg_sv = [state.svm_tracker.neg_sv;neg_sv_new];
    state.svm_tracker.neg_w = [state.svm_tracker.neg_w;ones(size(sv_new,1)-num_sv_pos_new,1)];
end
        
state.svm_tracker.pos_dis = state.svm_tracker.pos_dis + diag(inf*ones(size(state.svm_tracker.pos_dis,1),1));
state.svm_tracker.neg_dis = state.svm_tracker.neg_dis + diag(inf*ones(size(state.svm_tracker.neg_dis,1),1));
        
        
% compute real margin
pos2plane = -state.svm_tracker.pos_sv*state.svm_tracker.w';
neg2plane = -state.svm_tracker.neg_sv*state.svm_tracker.w';
state.svm_tracker.margin = (min(pos2plane) - max(neg2plane))/norm(state.svm_tracker.w);
        
% shrink svs
% check if to remove
if size(state.svm_tracker.pos_sv,1)+size(state.svm_tracker.neg_sv,1)>state.svm_tracker.B
    pos_score_sv = -(state.svm_tracker.pos_sv*state.svm_tracker.w'+state.svm_tracker.Bias);
    neg_score_sv = -(state.svm_tracker.neg_sv*state.svm_tracker.w'+state.svm_tracker.Bias);
    m_pos = abs(pos_score_sv) < state.svm_tracker.m2;
    m_neg = abs(neg_score_sv) < state.svm_tracker.m2;
            
    if sum(m_pos) > 0
        state.svm_tracker.pos_sv = state.svm_tracker.pos_sv(m_pos,:);
        state.svm_tracker.pos_w = state.svm_tracker.pos_w(m_pos,:);
        state.svm_tracker.pos_dis = state.svm_tracker.pos_dis(m_pos,m_pos);
    end

    if sum(m_neg)>0
        state.svm_tracker.neg_sv = state.svm_tracker.neg_sv(m_neg,:);
        state.svm_tracker.neg_w = state.svm_tracker.neg_w(m_neg,:);
        state.svm_tracker.neg_dis = state.svm_tracker.neg_dis(m_neg,m_neg);
    end
end
        
% check if to merge
while size(state.svm_tracker.pos_sv,1)+size(state.svm_tracker.neg_sv,1)>state.svm_tracker.B
    [mm_pos,idx_pos] = min(state.svm_tracker.pos_dis(:));
    [mm_neg,idx_neg] = min(state.svm_tracker.neg_dis(:));
            
    if mm_pos > mm_neg || size(state.svm_tracker.pos_sv,1) <= state.svm_tracker.B_p% merge negative samples
  
        [i,j] = ind2sub(size(state.svm_tracker.neg_dis),idx_neg);
        w_i= state.svm_tracker.neg_w(i);
        w_j= state.svm_tracker.neg_w(j);
        merge_sample = (w_i*state.svm_tracker.neg_sv(i,:)+w_j*state.svm_tracker.neg_sv(j,:))/(w_i+w_j);                
                
        state.svm_tracker.neg_sv([i,j],:) = []; state.svm_tracker.neg_sv(end+1,:) = merge_sample;
        state.svm_tracker.neg_w([i,j]) = []; state.svm_tracker.neg_w(end+1,1) = w_i + w_j;
                
        state.svm_tracker.neg_dis([i,j],:)=[]; state.svm_tracker.neg_dis(:,[i,j])=[];
        neg_dis_cro = pdist2(state.svm_tracker.neg_sv(1:end-1,:),merge_sample);
        state.svm_tracker.neg_dis = [state.svm_tracker.neg_dis, neg_dis_cro;neg_dis_cro',inf];                
    else
        [i,j] = ind2sub(size(state.svm_tracker.pos_dis),idx_pos);
        w_i= state.svm_tracker.pos_w(i);
        w_j= state.svm_tracker.pos_w(j);
        merge_sample = (w_i*state.svm_tracker.pos_sv(i,:)+w_j*state.svm_tracker.pos_sv(j,:))/(w_i+w_j);                

        state.svm_tracker.pos_sv([i,j],:) = []; state.svm_tracker.pos_sv(end+1,:) = merge_sample;
        state.svm_tracker.pos_w([i,j]) = []; state.svm_tracker.pos_w(end+1,1) = w_i + w_j;
                
        state.svm_tracker.pos_dis([i,j],:)=[]; state.svm_tracker.pos_dis(:,[i,j])=[];
        pos_dis_cro = pdist2(state.svm_tracker.pos_sv(1:end-1,:),merge_sample);
        state.svm_tracker.pos_dis = [state.svm_tracker.pos_dis, pos_dis_cro;pos_dis_cro',inf]; 
                
                
    end
            
end
        
% update experts
state.experts{end}.w = state.svm_tracker.w;
state.experts{end}.Bias = state.svm_tracker.Bias;
        
state.svm_tracker.update_count = state.svm_tracker.update_count + 1;
```
### [tracker_meem_update.m](meem/tracker_meem_update.m)
updates the meem tracker
```matlab
function [state, location, values] = tracker_meem_update(state, image, varargin)

    values = struct();

    state.time = state.time + 1;

	[I_scale, ~] = image_convert(image_create(image), 'rgb');

    % compute ROI and scale image
    if state.config.image_scale ~= 1
        I_scale = cv.resize(I_scale, state.config.image_scale, state.config.image_scale);
    end

    if state.config.padding > 0
        I_scale = padarray(I_scale, [state.config.padding, state.config.padding], 'replicate');
    end

    state.sampler.roi = rsz_rt(state.output, size(I_scale), state.config.search_roi, true);

    I_crop = I_scale(round(state.sampler.roi(2):state.sampler.roi(4)),round(state.sampler.roi(1):state.sampler.roi(3)),:);

    % compute feature images
    [BC, ~] = getFeatureRep(I_crop, state.config);

    % tracking part

    if mod(state.time, state.config.expert_update_interval) == 0 % svm_tracker.update_count >= config.update_count_thresh
        if numel(state.experts) < state.config.max_expert_sz
            state.svm_tracker.update_count = 0;
            state.experts{end}.snapshot = state.svm_tracker;
            state.experts{end+1} = state.experts{end};
        else
            state.svm_tracker.update_count = 0;
            state.experts{end}.snapshot = state.svm_tracker;
            state.experts(1:end-1) = state.experts(2:end);
        end
    end

    state = expertsDo(state, BC);

    if state.svm_tracker.confidence > state.config.svm_thresh
        state.output = state.svm_tracker.output;
    end

    % update svm classifier
    state.svm_tracker.temp_count = state.svm_tracker.temp_count + 1;

    if state.svm_tracker.confidence > state.config.svm_thresh %&& ~svm_tracker.failure
        train_mask = (state.sampler.costs < state.config.thresh_p) | (state.sampler.costs >= state.config.thresh_n);
        label = state.sampler.costs(train_mask) < state.config.thresh_p;

        skip_train = false;
        if state.svm_tracker.confidence > 1.0
            score_ = -(state.sampler.patterns_dt(train_mask,:) * state.svm_tracker.w' + state.svm_tracker.Bias);
            if prod(double(score_(label) > 1)) == 1 && prod(double(score_(~label)<1)) == 1
                skip_train = true;
            end
        end

        if ~skip_train
            costs = state.sampler.costs(train_mask);
            fuzzy_weight = ones(size(label));
            fuzzy_weight(~label) = 2*costs(~label)-1;
            state = updateSvmTracker(state, state.sampler.patterns_dt(train_mask,:), label, fuzzy_weight);
        end
    else
        state.svm_tracker.update_count = 0;
    end

    res = state.output;
    res(1:2) = res(1:2) - state.config.padding;
    location = res / state.config.image_scale;

end
```
### [tracker_meem_initialize.m](meem/tracker_meem_initialize.m)
initializes the meem tracker
```matlab
function [state, location, values] = tracker_meem_initialize(image, region, varargin)

warning('off','MATLAB:maxNumCompThreads:Deprecated');
maxNumCompThreads(1);% The rounding error due to using differen number of threads
                     % could cause different tracking results.  On most sequences,
                     % the difference is very small, while on some challenging
                     % sequences, the difference can be substantial due to
                     % "butterfly effects". Therefore, we suggest using
                     % Spatial Robustness Evaluation (SRE) to benchmark
                     % trackers.

defaults.search_roi = 2; % ratio of the search roi to tracking window
defaults.padding = 40; % for object out of border

defaults.debug = false;
defaults.verbose = false;
defaults.use_experts = true;
defaults.use_color = true;
defaults.use_raw_feat = false; % raw intensity feature value
defaults.use_iif = true; % use illumination invariant feature

defaults.svm_thresh = -0.7; % for detecting the tracking failure
defaults.max_expert_sz = 4;
defaults.expert_update_interval = 50;
defaults.update_count_thresh = 1;
defaults.entropy_score_winsize = 5;
defaults.expert_lambda = 10;
defaults.label_prior_sigma = 15;

defaults.hist_nbin = 32; % histogram bins for iif computation

defaults.thresh_p = 0.1; % IOU threshold for positive training samples
defaults.thresh_n = 0.5; % IOU threshold for negative ones

parameters = struct();

for i = 1:2:length(varargin)
    switch lower(varargin{i})
        case 'parameters'
            parameters = varargin{i+1};
        otherwise
            error(['Unknown switch ', varargin{i},'!']) ;
    end
end

% intialization
init_rect = round(region);
state.config = struct_merge(parameters, defaults);

[frame, ~] = image_convert(image_create(image), 'rgb');

thr_n = 5;
state.config.thr = (1/thr_n:1/thr_n:1-1/thr_n)*255;
state.config.fd = numel(state.config.thr);

% decide image scale and pixel step for sampling feature
% rescale raw input frames propoerly would save much computation
frame_min_width = 320;
trackwin_max_dimension = 64;
template_max_numel = 144;
frame_sz = size(frame);

if max(init_rect(3:4)) <= trackwin_max_dimension ||...
        frame_sz(2) <= frame_min_width
    state.config.image_scale = 1;
else
    min_scale = frame_min_width/frame_sz(2);
    state.config.image_scale = max(trackwin_max_dimension/max(init_rect(3:4)),min_scale);
end
wh_rescale = init_rect(3:4) * state.config.image_scale;
win_area = prod(wh_rescale);
state.config.ratio = (sqrt(template_max_numel/win_area));
template_sz = round(wh_rescale * state.config.ratio);
state.config.template_sz = template_sz([2 1]);

state.sampler = createSampler();
state.svm_tracker = createSvmTracker();
state.experts = {};

state.svm_tracker.output = init_rect * state.config.image_scale;
state.svm_tracker.output(1:2) = state.svm_tracker.output(1:2) + state.config.padding;
state.svm_tracker.output_exp = state.svm_tracker.output;

state.output = state.svm_tracker.output;

I_scale = frame;

% compute ROI and scale image
if state.config.image_scale ~= 1
    I_scale = cv.resize(I_scale, state.config.image_scale, state.config.image_scale);
end

if state.config.padding > 0
    I_scale = padarray(I_scale, [state.config.padding, state.config.padding], 'replicate');
end

state.sampler.roi = rsz_rt(state.svm_tracker.output, size(I_scale), 5 * state.config.search_roi, false);

I_crop = I_scale(round(state.sampler.roi(2):state.sampler.roi(4)),round(state.sampler.roi(1):state.sampler.roi(3)),:);

% compute feature images
[BC, ~] = getFeatureRep(I_crop, state.config);

% tracking part

state = initSampler(state, state.svm_tracker.output, BC, state.config);
train_mask = (state.sampler.costs < state.config.thresh_p) | (state.sampler.costs >= state.config.thresh_n);
label = state.sampler.costs(train_mask,1) < state.config.thresh_p;
fuzzy_weight = ones(size(label));
state = initSvmTracker(state, state.sampler.patterns_dt(train_mask,:), label, fuzzy_weight);

state.time = 0;

location = region;
values = struct();

end
```

### [updateTrackerExperts.m](visual-tracking-matlab/meem/updateTrackerExperts.m)
function to update the experts
```matlab
function updateTrackerExperts
global config
global svm_tracker
global experts

if numel(experts) < config.max_expert_sz
    svm_tracker.update_count = 0;
    experts{end}.snapshot = svm_tracker;
    experts{end+1} = experts{end};
else
    svm_tracker.update_count = 0;
    experts{end}.snapshot = svm_tracker;
    experts(1:end-1) = experts(2:end);
end
```

### [expertsDo.m](visual-tracking-matlab/meem/expertsDo.m)
- Calls the fuction `expertsDo()`

```matlab

function state = expertsDo(state, I_vf)
    % expertsDo - Do the experts
    %
    %   state = expertsDo(state, I_vf)
    %
    %   state - The state of the tracker
    %   I_vf - The current frame
    %
    %   Returns the updated state
    %
    %   See also: tracker_meem_initialize, tracker_meem_update
    %
roi_reg = state.sampler.roi; 
roi_reg(3:4) = state.sampler.roi(3:4)-state.sampler.roi(1:2);

feature_map = imresize(I_vf,state.config.ratio,'nearest');
ratio_x = size(I_vf,2)/size(feature_map,2);
ratio_y = size(I_vf,1)/size(feature_map,1);
patterns = im2colstep(feature_map,[state.sampler.template_size(1:2), size(I_vf,3)],[1, 1, size(I_vf,3)]);

x_sz = size(feature_map,2)-state.sampler.template_size(2)+1;
y_sz = size(feature_map,1)-state.sampler.template_size(1)+1;
[X, Y] = meshgrid(1:x_sz,1:y_sz);
temp = repmat(state.svm_tracker.output,[numel(X),1]);
temp(:,1) = (X(:)-1)*ratio_x + state.sampler.roi(1);
temp(:,2) = (Y(:)-1)*ratio_y + state.sampler.roi(2);

% select expert
label_prior = fspecial('gaussian',[y_sz,x_sz], state.config.label_prior_sigma);
label_prior_neg = ones(size(label_prior))/numel(label_prior);

% compute log likelihood and entropy
n = numel(state.experts);
score_temp = zeros(n,1);
rect_temp = zeros(n,4);

rad = 0.5 * min(state.sampler.template_size(1:2));

mask_temp = zeros(y_sz,x_sz);
idx_temp = [];
svm_scores = [];
svm_score = {};
svm_density = {};
peaks_collection = {};
peaks = zeros(n,2);
peaks_pool = [];

for i = 1:n
    % find the highest peak
    svm_score{i} = -(state.experts{i}.w*patterns+state.experts{i}.Bias);
    svm_density{i} = normcdf(svm_score{i},0,1).*label_prior(:)';
    [val, idx] = max(svm_density{i});
    best_rect = temp(idx,:);
    rect_temp(i,:) = best_rect;
    svm_scores(i) = svm_score{i}(idx);
    idx_temp(i) = idx;
    [r c] = ind2sub(size(mask_temp),idx);
    peaks(i,:) = [r c];
    
    % find the possible peaks
    
    density_map = reshape(svm_density{i},y_sz,[]);
    density_map = (density_map - min(density_map(:)))/(max(density_map(:)) - min(density_map(:)));
    mm = (imdilate(density_map,strel('square',round(rad))) == density_map) & density_map > 0.9;
    [rn cn] = ind2sub(size(mask_temp),find(mm));
    peaks_pool = cat(1,peaks_pool,[rn cn]);  
    peaks_collection{i} = [rn cn];
end

% merg peaks
peaks = mergePeaks(peaks,rad);
peaks_pool = mergePeaks(peaks_pool,rad);
mask_temp(sub2ind(size(mask_temp),round(peaks(:,1)),round(peaks(:,2)))) = 1;

for i = 1:n

    dis = pdist2(peaks_pool,peaks_collection{i});
    [rr, cc] = ind2sub([size(peaks_pool,1),size(peaks_collection{i},1)],find(dis < rad));
    [~, ia, ~] = unique(cc);
    peaks_temp = peaks_pool;
    peaks_temp(rr(ia),:) = peaks_collection{i}(cc(ia),:);
    mask = zeros(size(mask_temp));
    mask(sub2ind(size(mask_temp),round(peaks_temp(:,1)),round(peaks_temp(:,2)))) = 1;
    mask = mask>0;

    [loglik, ent] = getLogLikelihoodEntropy(svm_score{i}(mask(:)),label_prior(mask(:)),label_prior_neg(mask(:)));

    state.experts{i}.score(end+1) =  loglik - state.config.expert_lambda * ent;
    score_temp(i) = sum(state.experts{i}.score(max(end+1-state.config.entropy_score_winsize,1):end));    
end

%
state.svm_tracker.best_expert_idx = numel(score_temp);
if numel(score_temp) >= 2 && state.config.use_experts
    [~, idx] = max(score_temp(1:end-1));
    if score_temp(idx) > score_temp(end) && size(peaks,1) > 1
        state.experts{end}.score = state.experts{idx}.score;
        state.svm_tracker = state.experts{idx}.snapshot;
        state.svm_tracker.best_expert_idx = idx;

    end
end
state.svm_tracker.output = rect_temp(state.svm_tracker.best_expert_idx,:);
state.svm_tracker.confidence = svm_scores(state.svm_tracker.best_expert_idx);
state.svm_tracker.output_exp = rect_temp(end,:);
state.svm_tracker.confidence_exp = svm_scores(end);

% update training sample
% approximately 200 training samples
step = round(sqrt((y_sz*x_sz)/120));
mask_temp = zeros(y_sz,x_sz);
mask_temp(1:step:end,1:step:end) = 1;
mask_temp = mask_temp > 0;
state.sampler.patterns_dt = patterns(:,mask_temp(:))';
state.sampler.state_dt = temp(mask_temp(:),:);
state.sampler.costs = 1 - getIOU(state.sampler.state_dt,state.svm_tracker.output);
if min(state.sampler.costs)~=0
    state.sampler.state_dt = [state.sampler.state_dt; rect_temp(state.svm_tracker.best_expert_idx,:)];
    state.sampler.patterns_dt = [state.sampler.patterns_dt; patterns(:,idx_temp(state.svm_tracker.best_expert_idx))'];
    state.sampler.costs = [state.sampler.costs;0];
end

end

function merged_peaks = mergePeaks(peaks, rad)

dis_mat = pdist2(peaks,peaks) + diag(inf*ones(size(peaks,1),1));
while min(dis_mat(:)) < rad && size(peaks,1) > 1
    [~, idx] = min(dis_mat(:));
    [id1, id2] = ind2sub(size(dis_mat),idx);
    merged_peak = 0.5*(peaks(id1,:) + peaks(id2,:));
    peaks([id1 id2],:) = [];
    peaks = [peaks;merged_peak];
    dis_mat = pdist2(peaks,peaks) + diag(inf*ones(size(peaks,1),1));
end

merged_peaks = peaks;

end

```
### [getLogLikelihoodEntropy.m](meem/codegetLogLikelihoodEntropy)
calculates the log likelihood and entropy of a given state
```matlab
function [ll, entropy] = getLogLikelihoodEntropy (svm_score,label_prior,label_prior_neg)

num = numel(svm_score);

pos_score = normcdf(svm_score,0,1);
pos_score = pos_score.*label_prior(:)';

neg_score = 1 - pos_score;
neg_score = neg_score.*label_prior_neg(:)';
p_XY_Z = prod(repmat(neg_score(:),[1 num])+diag(pos_score - neg_score));
g_XY_Z = p_XY_Z/sum(p_XY_Z);
entropy = -g_XY_Z*log(g_XY_Z)';% in case g is 0
ll = log(max(p_XY_Z));
```

### [GetFeatureRep.m](meem/GetFeatureRep.m)
Computes the feature representation of the given state.
```matlab
 compute feature representation: mxnxd, d is the feature dimension
% decay factor and nbin is for the local histogram computation
if size(I,3) == 3 && config.use_color
    I = uint8(255*RGB2Lab(I));
elseif size(I,3) == 3
    I = rgb2gray(I);
end
fd = config.fd;
ksize = (1/config.ratio)*4;
if mod(ksize,2) == 0
    ksize = ksize + 1;
end
if config.use_iif
    F{1} = 255 - calcIIF(I(:,:,1),[ksize ksize],config.hist_nbin)';%IIF2(I(:,:,1)*255, hist_mtx1, nbin);%feature by pixel ordering
    if any(size(F{1}) ~= size(I(:,:, 1)))
        F{1} = F{1}';
    end
else
    F{1} = uint8(zeros([size(I,1),size(I,2)]));
end
F{2} = I(:,:,1);%gray image
if config.use_color
    F{3} = I(:,:,2);%color part
    F{4} = I(:,:,3);%color part
end
if config.use_raw_feat
    feat = double(reshape(cell2mat(F),size(F{1},1),size(F{1},2),[]))/255;
else
    if ~config.use_color
        feat = zeros([size(I(:,:,1)),2*config.fd]);
    else
        feat = zeros([size(I(:,:,1)),4*config.fd]);
    end
    for i = 1:numel(F)
        feat(:,:,(i-1)*fd+1:i*fd) = bsxfun(@gt,repmat(F{i},[1 1 fd]), reshape(config.thr,1,1,[]));
    end
end
F = double(reshape(cell2mat(F),size(F{1},1),size(F{1},2),[]));
```

 ### [calIIf.cpp](meem/calcIIF.cpp)

- Mex interface for IIF computation routine

### [RGB2Lab.m](meem/RGB2Lab.m)
Converts an RGB image to the L*a*b* color space.
```matlab
function [L,a,b] = RGB2Lab(R,G,B)
%RGB2LAB Convert an image from RGB to CIELAB
%
% function [L, a, b] = RGB2Lab(R, G, B)
% function [L, a, b] = RGB2Lab(I)
% function I = RGB2Lab(...)
%
% RGB2Lab takes red, green, and blue matrices, or a single M x N x 3 image, 
% and returns an image in the CIELAB color space.  RGB values can be
% either between 0 and 1 or between 0 and 255.  Values for L are in the
% range [0,1] while a and b are roughly in the range [0,1].  The
% output is of type double.
%
% This transform is based on ITU-R Recommendation BT.709 using the D65
% white point reference. The error in transforming RGB -> Lab -> RGB is
% approximately 10^-5.  
%
% See also LAB2RGB.

% By Mark Ruzon from C code by Yossi Rubner, 23 September 1997.
% Updated for MATLAB 5 28 January 1998.
% Updated for MATLAB 7 30 March 2009.

if nargin == 1
  B = double(R(:,:,3));
  G = double(R(:,:,2));
  R = double(R(:,:,1));
end

if max(max(R)) > 1.0 || max(max(G)) > 1.0 || max(max(B)) > 1.0
  R = double(R) / 255;
  G = double(G) / 255;
  B = double(B) / 255;
end

% Set a threshold
T = 0.008856;

[M, N] = size(R);
s = M * N;
RGB = [reshape(R,1,s); reshape(G,1,s); reshape(B,1,s)];

% RGB to XYZ
MAT = [0.412453 0.357580 0.180423;
       0.212671 0.715160 0.072169;
       0.019334 0.119193 0.950227];
XYZ = MAT * RGB;

% Normalize for D65 white point
X = XYZ(1,:) / 0.950456;
Y = XYZ(2,:);
Z = XYZ(3,:) / 1.088754;

XT = X > T;
YT = Y > T;
ZT = Z > T;

Y3 = Y.^(1/3); 

fX = XT .* X.^(1/3) + (~XT) .* (7.787 .* X + 16/116);
fY = YT .* Y3 + (~YT) .* (7.787 .* Y + 16/116);
fZ = ZT .* Z.^(1/3) + (~ZT) .* (7.787 .* Z + 16/116);

L = (reshape(YT .* (116 * Y3 - 16.0) + (~YT) .* (903.3 * Y), M, N))/100;
a = (reshape(500 * (fX - fY), M, N)*3+110)/220;
b = (reshape(200 * (fY - fZ), M, N)*3+110)/220;

if nargout < 2
  L = cat(3,L,a,b);
end
```