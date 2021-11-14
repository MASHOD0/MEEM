# [Base Implementation](https://github.com/lukacu/visual-tracking-matlab)

## File Structure Tree:
```
📦base_implementation
 ┗ 📂visual-tracking-matlab
 ┃ ┣ 📂ant
 ┃ ┃ ┣ 📂memory
 ┃ ┃ ┃ ┣ 📜create_kcf.m
 ┃ ┃ ┃ ┣ 📜fhog.m
 ┃ ┃ ┃ ┣ 📜gaussian_correlation.m
 ┃ ┃ ┃ ┣ 📜gaussian_shaped_labels.m
 ┃ ┃ ┃ ┣ 📜get_features.m
 ┃ ┃ ┃ ┣ 📜get_subwindow.m
 ┃ ┃ ┃ ┣ 📜gradients.cpp
 ┃ ┃ ┃ ┣ 📜linear_correlation.m
 ┃ ┃ ┃ ┣ 📜match_kcf.m
 ┃ ┃ ┃ ┣ 📜match_kcfs.m
 ┃ ┃ ┃ ┣ 📜match_ncc.m
 ┃ ┃ ┃ ┣ 📜memory_create.m
 ┃ ┃ ┃ ┣ 📜memory_draw.m
 ┃ ┃ ┃ ┣ 📜memory_match.m
 ┃ ┃ ┃ ┣ 📜memory_remove.m
 ┃ ┃ ┃ ┣ 📜memory_update.m
 ┃ ┃ ┃ ┣ 📜polynomial_correlation.m
 ┃ ┃ ┃ ┣ 📜sse.hpp
 ┃ ┃ ┃ ┗ 📜wrappers.hpp
 ┃ ┃ ┣ 📂segmentation
 ┃ ┃ ┃ ┣ 📜segmentation_create.m
 ┃ ┃ ┃ ┣ 📜segmentation_draw.m
 ┃ ┃ ┃ ┣ 📜segmentation_generate.m
 ┃ ┃ ┃ ┣ 📜segmentation_get_threshold.m
 ┃ ┃ ┃ ┣ 📜segmentation_process_mask.m
 ┃ ┃ ┃ ┣ 📜segmentation_sample.m
 ┃ ┃ ┃ ┗ 📜segmentation_update.m
 ┃ ┃ ┣ 📜ant_parameters.m
 ┃ ┃ ┣ 📜compute_coverage.m
 ┃ ┃ ┣ 📜compute_shift.m
 ┃ ┃ ┣ 📜fit_rectangle.m
 ┃ ┃ ┣ 📜README.md
 ┃ ┃ ┣ 📜tracker_ant_draw.m
 ┃ ┃ ┣ 📜tracker_ant_initialize.m
 ┃ ┃ ┗ 📜tracker_ant_update.m
 ┃ ┣ 📂common
 ┃ ┃ ┣ 📂parts
 ┃ ┃ ┃ ┣ 📜partassemble.cpp
 ┃ ┃ ┃ ┣ 📜partcompare.cpp
 ┃ ┃ ┃ ┣ 📜parts_add.m
 ┃ ┃ ┃ ┣ 📜parts_bounds.m
 ┃ ┃ ┃ ┣ 📜parts_create.m
 ┃ ┃ ┃ ┣ 📜parts_draw.m
 ┃ ┃ ┃ ┣ 📜parts_history.m
 ┃ ┃ ┃ ┣ 📜parts_mask.m
 ┃ ┃ ┃ ┣ 📜parts_match_ce.m
 ┃ ┃ ┃ ┣ 📜parts_match_icm.m
 ┃ ┃ ┃ ┣ 📜parts_merge.m
 ┃ ┃ ┃ ┣ 📜parts_mode.m
 ┃ ┃ ┃ ┣ 📜parts_push.m
 ┃ ┃ ┃ ┣ 📜parts_remove.m
 ┃ ┃ ┃ ┣ 📜parts_responses.m
 ┃ ┃ ┃ ┣ 📜parts_size.m
 ┃ ┃ ┃ ┗ 📜patches.cpp
 ┃ ┃ ┣ 📜affparam2geom.m
 ┃ ┃ ┣ 📜affparam2mat.m
 ┃ ┃ ┣ 📜affparaminv.m
 ┃ ┃ ┣ 📜affwarpimg.m
 ┃ ┃ ┣ 📜apply_transformation.m
 ┃ ┃ ┣ 📜c.m
 ┃ ┃ ┣ 📜calculate_overlap.m
 ┃ ┃ ┣ 📜catstruct.m
 ┃ ┃ ┣ 📜compile_mex.m
 ┃ ┃ ┣ 📜gauss.m
 ┃ ┃ ┣ 📜gaussderiv.m
 ┃ ┃ ┣ 📜hann.m
 ┃ ┃ ┣ 📜image_convert.m
 ┃ ┃ ┣ 📜image_create.m
 ┃ ┃ ┣ 📜image_crop.m
 ┃ ┃ ┣ 📜image_resize.m
 ┃ ┃ ┣ 📜image_size.m
 ┃ ┃ ┣ 📜immask.m
 ┃ ┃ ┣ 📜interp2.cpp
 ┃ ┃ ┣ 📜is_octave.m
 ┃ ┃ ┣ 📜kalman_update.m
 ┃ ┃ ┣ 📜normalize.m
 ┃ ┃ ┣ 📜patch_operation.m
 ┃ ┃ ┣ 📜plotc.m
 ┃ ┃ ┣ 📜plot_color_cloud.m
 ┃ ┃ ┣ 📜points2mask.m
 ┃ ┃ ┣ 📜points2rect.m
 ┃ ┃ ┣ 📜poly2bb.m
 ┃ ┃ ┣ 📜polydist.m
 ┃ ┃ ┣ 📜polygon_operation.m
 ┃ ┃ ┣ 📜polyoverlap.m
 ┃ ┃ ┣ 📜print_structure.m
 ┃ ┃ ┣ 📜rect2points.m
 ┃ ┃ ┣ 📜rectangle2poly.m
 ┃ ┃ ┣ 📜rectangle_operation.m
 ┃ ┃ ┣ 📜region2poly.m
 ┃ ┃ ┣ 📜sample_gaussian.m
 ┃ ┃ ┣ 📜scale.m
 ┃ ┃ ┣ 📜scalemax.m
 ┃ ┃ ┣ 📜scan_directory.m
 ┃ ┃ ┣ 📜sfigure.m
 ┃ ┃ ┣ 📜struct_merge.m
 ┃ ┃ ┣ 📜timer_create.m
 ┃ ┃ ┣ 📜timer_push.m
 ┃ ┃ ┣ 📜track.m
 ┃ ┃ ┣ 📜trax.m
 ┃ ┃ ┣ 📜wcov.m
 ┃ ┃ ┣ 📜wmean.m
 ┃ ┃ ┗ 📜wtransform.m
 ┃ ┣ 📂ivt
 ┃ ┃ ┣ 📜esterrfunc.m
 ┃ ┃ ┣ 📜estwarp_condens.m
 ┃ ┃ ┣ 📜estwarp_grad.m
 ┃ ┃ ┣ 📜estwarp_greedy.m
 ┃ ┃ ┣ 📜hall.m
 ┃ ┃ ┣ 📜README.md
 ┃ ┃ ┣ 📜sklm.m
 ┃ ┃ ┣ 📜tracker_ivt_initialize.m
 ┃ ┃ ┣ 📜tracker_ivt_update.m
 ┃ ┃ ┗ 📜warpimg.m
 ┃ ┣ 📂l1apg
 ┃ ┃ ┣ 📜aff2image.m
 ┃ ┃ ┣ 📜APGLASSOup.m
 ┃ ┃ ┣ 📜corner2image.m
 ┃ ┃ ┣ 📜corners2affine.m
 ┃ ┃ ┣ 📜crop_candidates.m
 ┃ ┃ ┣ 📜draw_sample.m
 ┃ ┃ ┣ 📜images_angle.m
 ┃ ┃ ┣ 📜imgaffine.c
 ┃ ┃ ┣ 📜InitTemplates.m
 ┃ ┃ ┣ 📜normalizeTemplates.m
 ┃ ┃ ┣ 📜README.md
 ┃ ┃ ┣ 📜resample.m
 ┃ ┃ ┣ 📜tracker_l1apg_initialize.m
 ┃ ┃ ┣ 📜tracker_l1apg_update.m
 ┃ ┃ ┗ 📜whitening.m
 ┃ ┣ 📂lgt
 ┃ ┃ ┣ 📂modalities
 ┃ ┃ ┃ ┣ 📜modalities_create.m
 ┃ ┃ ┃ ┣ 📜modalities_draw.m
 ┃ ┃ ┃ ┣ 📜modalities_sample.m
 ┃ ┃ ┃ ┗ 📜modalities_update.m
 ┃ ┃ ┣ 📜cone_kernel.m
 ┃ ┃ ┣ 📜gauss_kernel.m
 ┃ ┃ ┣ 📜lgt_parameters.m
 ┃ ┃ ┣ 📜README.md
 ┃ ┃ ┣ 📜sample_probability_map.m
 ┃ ┃ ┣ 📜tracker_lgt_draw.m
 ┃ ┃ ┣ 📜tracker_lgt_initialize.m
 ┃ ┃ ┗ 📜tracker_lgt_update.m
 ┃ ┣ 📂meem
 ┃ ┃ ┣ 📜calcIIF.cpp
 ┃ ┃ ┣ 📜calcIIF.m
 ┃ ┃ ┣ 📜createSampler.m
 ┃ ┃ ┣ 📜createSvmTracker.m
 ┃ ┃ ┣ 📜expertsDo.m
 ┃ ┃ ┣ 📜getFeatureRep.m
 ┃ ┃ ┣ 📜getIOU.m
 ┃ ┃ ┣ 📜getLogLikelihoodEntropy.m
 ┃ ┃ ┣ 📜im2colstep.c
 ┃ ┃ ┣ 📜im2colstep.m
 ┃ ┃ ┣ 📜initSampler.m
 ┃ ┃ ┣ 📜initSvmTracker.m
 ┃ ┃ ┣ 📜README.md
 ┃ ┃ ┣ 📜resample.m
 ┃ ┃ ┣ 📜RGB2Lab.m
 ┃ ┃ ┣ 📜rsz_rt.m
 ┃ ┃ ┣ 📜tracker_meem_initialize.m
 ┃ ┃ ┣ 📜tracker_meem_update.m
 ┃ ┃ ┣ 📜updateSample.m
 ┃ ┃ ┣ 📜updateSvmTracker.m
 ┃ ┃ ┗ 📜updateTrackerExperts.m
 ┃ ┣ 📜.git
 ┃ ┃     - Git files
 ┃ ┣ 📜.gitignore
 ┃ ┃     - Git ignore file
 ┃ ┃    
 ┃ ┣ 📜compile_native.m
 ┃ ┃    - Compile native code
 ┃ ┃      (requires mex)
 ┃ ┃    
 ┃ ┣ 📜README.md
 ┃ ┣ 📜run_ant.m
 ┃ ┃   - Runs ant
 ┃ ┃   ANT - Visual tracking using anchor templates
 ┃ ┣ 📜run_ivt.m 
 ┃ ┃  - Runs ivt
 ┃ ┃  IVT - Incremental Learning for Robust Visual Tracking
 ┃ ┣ 📜run_l1apg.m
 ┃ ┃  - Runs l1apg
 ┃ ┃  L1APG - L1 Tracking using acclerated proximal gradient
 ┃ ┣ 📜run_lgt.m
 ┃ ┃  - Runs lgt
 ┃ ┃  LGT - Local - Global Appearance Tracker
 ┃ ┗ 📜run_meem.m
      - Runs meem tracker
```