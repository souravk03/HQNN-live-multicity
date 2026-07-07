// diagram.js — pipeline SVG + node glow animation
// ===== Pipeline flowchart (matches design) — driven by live node events =====
const PIPE_SVG=`<svg viewBox="0 0 2451.0 870.9000000000001" xmlns="http://www.w3.org/2000/svg" class="pipe">
<defs><marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3.2" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,3.2 L0,6.4 Z" fill="#5b6b80"/></marker><marker id="arrH" markerWidth="13" markerHeight="13" refX="8" refY="3.8" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,3.8 L0,7.6 Z" fill="#b8531a"/></marker></defs>
<rect class="quad quad-data" x="30.5" y="1.0" width="915.0" height="357.98" rx="16"/>
<text class="qlabel qlabel-data" x="46.5" y="27.0">Data Engineering</text>
<rect class="quad quad-tune" x="1262.0" y="1.0" width="1063.0" height="406.0" rx="16"/>
<text class="qlabel qlabel-tune" x="1278.0" y="27.0">Hyperparameter Tuning</text>
<rect class="quad quad-train" x="1262.0" y="486.9000000000001" width="798.0" height="242.0" rx="16"/>
<text class="qlabel qlabel-train" x="1278.0" y="512.9000000000001">Full Training</text>
<rect class="quad quad-daily" x="30.0" y="486.9000000000001" width="1160.0" height="314.0" rx="16"/>
<text class="qlabel qlabel-daily" x="46.0" y="512.9000000000001">Daily Forecast</text>
<path id="e_start_download" class="edge" d="M 244.0 78 L 276.0 78" marker-end="url(#arr)"/>
<path id="e_download_features" class="edge" d="M 424.0 78 L 456.0 78" marker-end="url(#arr)"/>
<path id="e_features_split" class="edge" d="M 604.0 78 L 636.0 78" marker-end="url(#arr)"/>
<path id="e_features_feat_lag" class="edge feat-link" d="M 530 103.0 L 530 166.3 L 197.0 166.3 L 197.0 183.3"/>
<path id="e_feat_lag_feat_roll" class="edge feat-link" d="M 271.0 211.3 L 402.0 211.3"/>
<path id="e_feat_roll_feat_diff" class="edge feat-link" d="M 550.0 211.3 L 681.0 211.3"/>
<path id="e_feat_diff_feat_cross" class="edge feat-link" d="M 755.0 236.3 L 755.0 265.42"/>
<path id="e_feat_cross_feat_cyc" class="edge feat-link" d="M 681.0 290.42 L 550.0 290.42"/>
<path id="e_feat_cyc_feat_anom" class="edge feat-link" d="M 402.0 290.42 L 271.0 290.42"/>
<path id="e_feat_anom_features" class="edge feat-link" d="M 123.0 290.42 L 79.82000000000001 290.42 L 79.82000000000001 127.0 L 500 127.0 L 500 107.0"/>
<path id="e_split_tune_start" class="edge handoff" d="M 784.0 78 L 1043.0 78 L 1043.0 78 L 1302.0 78" marker-end="url(#arr)"/>
<text class="elbl handoff-lbl" x="1043.0" y="66">→ tuning</text>
<path id="e_tune_start_t_tgt" class="edge" d="M 1450.0 78 L 1482.0 78" marker-end="url(#arr)"/>
<path id="e_t_tgt_t_mdl" class="edge" d="M 1630.0 78 L 1662.0 78" marker-end="url(#arr)"/>
<path id="e_t_mdl_optuna" class="edge" d="M 1810.0 78 L 1842.0 78" marker-end="url(#arr)"/>
<path id="e_optuna_sample" class="edge" d="M 1990.0 78 L 2022.0 78" marker-end="url(#arr)"/>
<path id="e_sample_cv" class="edge" d="M 2096.0 103.0 L 2096.0 139.0" marker-end="url(#arr)"/>
<path id="e_cv_fit" class="edge" d="M 2022.0 164 L 1990.0 164" marker-end="url(#arr)"/>
<path id="e_fit_buildseq" class="edge" d="M 1842.0 164 L 1810.0 164" marker-end="url(#arr)"/>
<path id="e_buildseq_quicktrain" class="edge" d="M 1662.0 164 L 1630.0 164" marker-end="url(#arr)"/>
<path id="e_quicktrain_valmse" class="edge" d="M 1482.0 164 L 1450.0 164" marker-end="url(#arr)"/>
<path id="e_valmse_report" class="edge" d="M 1376.0 189.0 L 1376.0 225.0" marker-end="url(#arr)"/>
<path id="e_report_prune" class="edge" d="M 1450.0 250 L 1482.0 250" marker-end="url(#arr)"/>
<path id="e_prune_morefolds" class="edge" d="M 1630.0 250 L 1662.0 250" marker-end="url(#arr)"/>
<text class="elbl" x="1646.0" y="243.0">No</text>
<path id="e_morefolds_avgmse" class="edge" d="M 1810.0 250 L 1842.0 250" marker-end="url(#arr)"/>
<text class="elbl" x="1826.0" y="243.0">No</text>
<path id="e_avgmse_moretrials" class="edge" d="M 1990.0 250 L 2022.0 250" marker-end="url(#arr)"/>
<path id="e_moretrials_sample" class="edge" d="M 2170.0 250 L 2198.0 250 L 2198.0 60 L 2174.0 60" marker-end="url(#arr)"/>
<text class="elbl" x="2210.0" y="242">Yes</text>
<path id="e_moretrials_selbest" class="edge" d="M 2096.0 281.0 L 2096.0 311.0" marker-end="url(#arr)"/>
<text class="elbl" x="2109.0" y="296.0">No</text>
<path id="e_selbest_moremodels" class="edge" d="M 2022.0 336 L 1990.0 336" marker-end="url(#arr)"/>
<path id="e_moremodels_moretgts" class="edge" d="M 1842.0 336 L 1810.0 336" marker-end="url(#arr)"/>
<text class="elbl" x="1826.0" y="329.0">No</text>
<path id="e_moretgts_writehp" class="edge" d="M 1662.0 336 L 1630.0 336" marker-end="url(#arr)"/>
<text class="elbl" x="1646.0" y="329.0">No</text>
<path id="e_moremodels_sample" class="edge" d="M 1916.0 367.0 L 1916.0 383.0 L 2224.0 383.0 L 2224.0 72 L 2174.0 72" marker-end="url(#arr)"/>
<text class="elbl" x="1932.0" y="394.0">Yes</text>
<path id="e_moretgts_sample" class="edge" d="M 1736.0 367.0 L 1736.0 397.0 L 2250.0 397.0 L 2250.0 84 L 2174.0 84" marker-end="url(#arr)"/>
<text class="elbl" x="1748.0" y="382.0">Yes</text>
<path id="e_prune_sample" class="edge" d="M 1556.0 281.0 L 1556.0 295.0 L 2276.0 295.0 L 2276.0 96 L 2174.0 96" marker-end="url(#arr)"/>
<text class="elbl" x="1578.0" y="288.0">Yes</text>
<path id="e_morefolds_cv" class="edge" d="M 1736.0 219.0 L 1736.0 205.0 L 2070.0 205.0 L 2070.0 193.0" marker-end="url(#arr)"/>
<text class="elbl" x="2100.0" y="198.0">Yes</text>
<path id="e_writehp_f_tgt" class="edge handoff" d="M 1556.0 361.0 L 1556.0 449.95000000000005 L 1736.0 449.95000000000005 L 1736.0 538.9000000000001" marker-end="url(#arr)"/>
<text class="elbl handoff-lbl" x="1736.0" y="437.95000000000005">→ training</text>
<path id="e_f_tgt_f_mdl" class="edge" d="M 1810.0 563.9000000000001 L 1842.0 563.9000000000001" marker-end="url(#arr)"/>
<path id="e_f_mdl_epoch" class="edge" d="M 1916.0 588.9000000000001 L 1916.0 624.9000000000001" marker-end="url(#arr)"/>
<path id="e_epoch_early" class="edge" d="M 1842.0 649.9000000000001 L 1810.0 649.9000000000001" marker-end="url(#arr)"/>
<path id="e_early_savew" class="edge" d="M 1662.0 649.9000000000001 L 1630.0 649.9000000000001" marker-end="url(#arr)"/>
<text class="elbl" x="1646.0" y="642.9000000000001">done</text>
<path id="e_savew_meta" class="edge" d="M 1482.0 649.9000000000001 L 1450.0 649.9000000000001" marker-end="url(#arr)"/>
<path id="e_early_epoch" class="edge" d="M 1736.0 680.9000000000001 L 1736.0 694.9000000000001 L 1916.0 694.9000000000001 L 1916.0 674.9000000000001" marker-end="url(#arr)"/>
<text class="elbl" x="1826.0" y="705.9000000000001">no - next epoch</text>
<path id="e_epoch_f_mdl" class="edge" d="M 1990.0 649.9000000000001 L 2034.0 649.9000000000001 L 2034.0 563.9000000000001 L 1990.0 563.9000000000001" marker-end="url(#arr)"/>
<text class="elbl" x="1988.0" y="606.9000000000001">next model ×6</text>
<path id="e_f_mdl_f_tgt" class="edge" d="M 1916.0 538.9000000000001 L 1916.0 520.9000000000001 L 1766.0 520.9000000000001 L 1766.0 534.9000000000001" marker-end="url(#arr)"/>
<text class="elbl" x="1841.0" y="513.9000000000001">next target ×3</text>
<path id="e_meta_beginday" class="edge handoff" d="M 1376.0 624.9000000000001 L 1376.0 563.9000000000001 L 1148.0 563.9000000000001" marker-end="url(#arr)"/>
<text class="elbl handoff-lbl" x="1260.0" y="554.9000000000001">→ daily</text>
<path id="e_beginday_forecast" class="edge" d="M 996.0 563.9000000000001 L 964.0 563.9000000000001" marker-end="url(#arr)"/>
<path id="e_forecast_verify" class="edge" d="M 816.0 563.9000000000001 L 784.0 563.9000000000001" marker-end="url(#arr)"/>
<path id="e_verify_agree" class="edge" d="M 636.0 563.9000000000001 L 604.0 563.9000000000001" marker-end="url(#arr)"/>
<path id="e_agree_seventh" class="edge" d="M 456.0 563.9000000000001 L 424.0 563.9000000000001" marker-end="url(#arr)"/>
<text class="elbl" x="440.0" y="556.9000000000001">Yes</text>
<path id="e_seventh_finetune" class="edge" d="M 276.0 563.9000000000001 L 244.0 563.9000000000001" marker-end="url(#arr)"/>
<text class="elbl" x="260.0" y="556.9000000000001">No</text>
<path id="e_agree_skipupd" class="edge" d="M 530 594.9000000000001 L 530 624.9000000000001" marker-end="url(#arr)"/>
<text class="elbl" x="543.0" y="609.9000000000001">No</text>
<path id="e_seventh_retrain" class="edge" d="M 350 594.9000000000001 L 350 624.9000000000001" marker-end="url(#arr)"/>
<text class="elbl" x="363.0" y="609.9000000000001">Yes</text>
<path id="e_retrain_savew2" class="edge" d="M 276.0 649.9000000000001 L 244.0 649.9000000000001" marker-end="url(#arr)"/>
<text class="elbl" x="260.0" y="619.9000000000001">Retrained</text>
<path id="e_finetune_savew2" class="edge" d="M 170 588.9000000000001 L 170 624.9000000000001" marker-end="url(#arr)"/>
<path id="e_savew2_logm" class="edge" d="M 170 674.9000000000001 L 170 710.9000000000001" marker-end="url(#arr)"/>
<path id="e_logm_moredays" class="edge" d="M 244.0 735.9000000000001 L 456.0 735.9000000000001" marker-end="url(#arr)"/>
<path id="e_moredays_skip" class="edge" d="M 604.0 735.9000000000001 L 996.0 735.9000000000001" marker-end="url(#arr)"/>
<text class="elbl" x="800.0" y="728.9000000000001">No</text>
<path id="e_skipupd_logm" class="edge" d="M 530 674.9000000000001 L 530 692.9000000000001 L 274.0 692.9000000000001 L 274.0 735.9000000000001 L 244.0 735.9000000000001" marker-end="url(#arr)"/>
<text class="elbl" x="464" y="684.9000000000001">Skip log</text>
<path id="e_moredays_beginday" class="edge" d="M 604.0 735.9000000000001 L 634.0 735.9000000000001 L 634.0 622.9000000000001 L 1070 622.9000000000001 L 1070 588.9000000000001" marker-end="url(#arr)"/>
<text class="elbl" x="974" y="613.9000000000001">yes · next day ↻</text>
<g class="fn node" id="nd_start" data-id="start"><rect x="96.0" y="53.0" width="148" height="50" rx="24" class="se lane-data"/><text class="lbl"><tspan x="170" y="78">Start</tspan></text></g>
<g class="fn node" id="nd_download" data-id="download"><rect x="276.0" y="53.0" width="148" height="50" rx="9" class="bx lane-data"/><text class="lbl"><tspan x="350" y="78">Download Data</tspan></text></g>
<g class="fn node" id="nd_features" data-id="features"><rect x="456.0" y="53.0" width="148" height="50" rx="9" class="bx lane-data"/><text class="lbl"><tspan x="530" y="78">Engineer 83 Features</tspan></text></g>
<g class="fn node" id="nd_split" data-id="split"><rect x="636.0" y="53.0" width="148" height="50" rx="9" class="bx lane-data"/><text class="lbl"><tspan x="710" y="78">Split Train / Forward</tspan></text></g>
<g class="fn node feat-node" id="nd_feat_lag" data-id="feat_lag"><rect x="70.5" y="182.74" width="253.0" height="57.120000000000005" rx="8" class="featbx lane-data"/><text class="featlbl"><tspan x="197.0" y="211.3">Lag features</tspan></text></g>
<g class="fn node feat-node" id="nd_feat_roll" data-id="feat_roll"><rect x="349.5" y="182.74" width="253.0" height="57.120000000000005" rx="8" class="featbx lane-data"/><text class="featlbl"><tspan x="476.0" y="211.3">Rolling statistics</tspan></text></g>
<g class="fn node feat-node" id="nd_feat_diff" data-id="feat_diff"><rect x="628.5" y="182.74" width="253.0" height="57.120000000000005" rx="8" class="featbx lane-data"/><text class="featlbl"><tspan x="755.0" y="204.3">Differences &amp;</tspan><tspan x="755.0" y="217.3">tendencies</tspan></text></g>
<g class="fn node feat-node" id="nd_feat_anom" data-id="feat_anom"><rect x="70.5" y="261.86" width="253.0" height="57.120000000000005" rx="8" class="featbx lane-data"/><text class="featlbl"><tspan x="197.0" y="290.42">Anomalies</tspan></text></g>
<g class="fn node feat-node" id="nd_feat_cyc" data-id="feat_cyc"><rect x="349.5" y="261.86" width="253.0" height="57.120000000000005" rx="8" class="featbx lane-data"/><text class="featlbl"><tspan x="476.0" y="290.42">Cyclical encodings</tspan></text></g>
<g class="fn node feat-node" id="nd_feat_cross" data-id="feat_cross"><rect x="628.5" y="261.86" width="253.0" height="57.120000000000005" rx="8" class="featbx lane-data"/><text class="featlbl"><tspan x="755.0" y="283.42">Cross-variable</tspan><tspan x="755.0" y="296.42">features</tspan></text></g>
<g class="fn node" id="nd_tune_start" data-id="tune_start"><rect x="1302.0" y="53.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="1376.0" y="78">Start Tuning</tspan></text></g>
<g class="fn node" id="nd_t_tgt" data-id="t_tgt"><rect x="1482.0" y="53.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="1556.0" y="78">For Each Target ×3</tspan></text></g>
<g class="fn node" id="nd_t_mdl" data-id="t_mdl"><rect x="1662.0" y="53.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="1736.0" y="78">For Each Model ×6</tspan></text></g>
<g class="fn node" id="nd_optuna" data-id="optuna"><rect x="1842.0" y="53.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="1916.0" y="78">Create Optuna Study</tspan></text></g>
<g class="fn node" id="nd_sample" data-id="sample"><rect x="2022.0" y="53.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="2096.0" y="70">Sample Hyper-</tspan><tspan x="2096.0" y="84">Parameters</tspan></text></g>
<g class="fn node" id="nd_cv" data-id="cv"><rect x="2022.0" y="139.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="2096.0" y="164">For each CV ×3</tspan></text></g>
<g class="fn node" id="nd_fit" data-id="fit"><rect x="1842.0" y="139.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="1916.0" y="164">Fit Scalar (Fold)</tspan></text></g>
<g class="fn node" id="nd_buildseq" data-id="buildseq"><rect x="1662.0" y="139.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="1736.0" y="164">Build Sequence</tspan></text></g>
<g class="fn node" id="nd_quicktrain" data-id="quicktrain"><rect x="1482.0" y="139.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="1556.0" y="164">Quick-Train Model</tspan></text></g>
<g class="fn node" id="nd_valmse" data-id="valmse"><rect x="1302.0" y="139.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="1376.0" y="164">Validate → MSE</tspan></text></g>
<g class="fn node" id="nd_report" data-id="report"><rect x="1302.0" y="225.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="1376.0" y="250">Report to Optuna</tspan></text></g>
<g class="fn node" id="nd_prune" data-id="prune"><polygon points="1556.0,219.0 1630.0,250 1556.0,281.0 1482.0,250" class="dia lane-tune"/><text class="lbl"><tspan x="1556.0" y="250">Prune Trial?</tspan></text></g>
<g class="fn node" id="nd_morefolds" data-id="morefolds"><polygon points="1736.0,219.0 1810.0,250 1736.0,281.0 1662.0,250" class="dia lane-tune"/><text class="lbl"><tspan x="1736.0" y="250">More Folds?</tspan></text></g>
<g class="fn node" id="nd_avgmse" data-id="avgmse"><rect x="1842.0" y="225.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="1916.0" y="250">Average Fold MSE</tspan></text></g>
<g class="fn node" id="nd_moretrials" data-id="moretrials"><polygon points="2096.0,219.0 2170.0,250 2096.0,281.0 2022.0,250" class="dia lane-tune"/><text class="lbl"><tspan x="2096.0" y="250">More Trials?</tspan></text></g>
<g class="fn node" id="nd_selbest" data-id="selbest"><rect x="2022.0" y="311.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="2096.0" y="328">Select Best</tspan><tspan x="2096.0" y="342">Parameters</tspan></text></g>
<g class="fn node" id="nd_moremodels" data-id="moremodels"><polygon points="1916.0,305.0 1990.0,336 1916.0,367.0 1842.0,336" class="dia lane-tune"/><text class="lbl"><tspan x="1916.0" y="336">More Models?</tspan></text></g>
<g class="fn node" id="nd_moretgts" data-id="moretgts"><polygon points="1736.0,305.0 1810.0,336 1736.0,367.0 1662.0,336" class="dia lane-tune"/><text class="lbl"><tspan x="1736.0" y="336">More Targets?</tspan></text></g>
<g class="fn node" id="nd_writehp" data-id="writehp"><rect x="1482.0" y="311.0" width="148" height="50" rx="9" class="bx lane-tune"/><text class="lbl"><tspan x="1556.0" y="336">Write best_hParams</tspan></text></g>
<g class="fn node" id="nd_f_tgt" data-id="f_tgt"><rect x="1662.0" y="538.9000000000001" width="148" height="50" rx="9" class="bx lane-train"/><text class="lbl"><tspan x="1736.0" y="563.9000000000001">For Each Target ×3</tspan></text></g>
<g class="fn node" id="nd_f_mdl" data-id="f_mdl"><rect x="1842.0" y="538.9000000000001" width="148" height="50" rx="9" class="bx lane-train"/><text class="lbl"><tspan x="1916.0" y="563.9000000000001">For Each Model ×6</tspan></text></g>
<g class="fn node" id="nd_epoch" data-id="epoch"><rect x="1842.0" y="624.9000000000001" width="148" height="50" rx="9" class="bx lane-train"/><text class="lbl"><tspan x="1916.0" y="649.9000000000001">Train Epoch</tspan></text></g>
<g class="fn node" id="nd_early" data-id="early"><polygon points="1736.0,618.9000000000001 1810.0,649.9000000000001 1736.0,680.9000000000001 1662.0,649.9000000000001" class="dia lane-train"/><text class="lbl"><tspan x="1736.0" y="649.9000000000001">Early Stop?</tspan></text></g>
<g class="fn node" id="nd_savew" data-id="savew"><rect x="1482.0" y="624.9000000000001" width="148" height="50" rx="9" class="bx lane-train"/><text class="lbl"><tspan x="1556.0" y="649.9000000000001">Save Best Weights</tspan></text></g>
<g class="fn node" id="nd_meta" data-id="meta"><rect x="1302.0" y="624.9000000000001" width="148" height="50" rx="9" class="bx lane-train"/><text class="lbl"><tspan x="1376.0" y="649.9000000000001">Write Metadata</tspan></text></g>
<g class="fn node" id="nd_beginday" data-id="beginday"><rect x="996.0" y="538.9000000000001" width="148" height="50" rx="9" class="bx lane-daily"/><text class="lbl"><tspan x="1070" y="563.9000000000001">Begin Forward Day</tspan></text></g>
<g class="fn node" id="nd_forecast" data-id="forecast"><rect x="816.0" y="538.9000000000001" width="148" height="50" rx="9" class="bx lane-daily"/><text class="lbl"><tspan x="890" y="563.9000000000001">Forecast next 15 days</tspan></text></g>
<g class="fn node" id="nd_verify" data-id="verify"><rect x="636.0" y="538.9000000000001" width="148" height="50" rx="9" class="bx lane-daily"/><text class="lbl"><tspan x="710" y="563.9000000000001">Verify vs Source</tspan></text></g>
<g class="fn node" id="nd_agree" data-id="agree"><polygon points="530,532.9000000000001 604.0,563.9000000000001 530,594.9000000000001 456.0,563.9000000000001" class="dia lane-daily"/><text class="lbl"><tspan x="530" y="563.9000000000001">Source Agrees?</tspan></text></g>
<g class="fn node" id="nd_seventh" data-id="seventh"><polygon points="350,532.9000000000001 424.0,563.9000000000001 350,594.9000000000001 276.0,563.9000000000001" class="dia lane-daily"/><text class="lbl"><tspan x="350" y="555.9000000000001">7th Verified Day /</tspan><tspan x="350" y="569.9000000000001">1st of Month</tspan></text></g>
<g class="fn node" id="nd_finetune" data-id="finetune"><rect x="96.0" y="538.9000000000001" width="148" height="50" rx="9" class="bx lane-daily"/><text class="lbl"><tspan x="170" y="563.9000000000001">Fine-Tune Weights</tspan></text></g>
<g class="fn node" id="nd_skipupd" data-id="skipupd"><rect x="456.0" y="624.9000000000001" width="148" height="50" rx="9" class="bx lane-daily"/><text class="lbl"><tspan x="530" y="649.9000000000001">Skip Update</tspan></text></g>
<g class="fn node" id="nd_retrain" data-id="retrain"><rect x="276.0" y="624.9000000000001" width="148" height="50" rx="9" class="bx lane-daily"/><text class="lbl"><tspan x="350" y="649.9000000000001">Full Retrain ×18</tspan></text></g>
<g class="fn node" id="nd_savew2" data-id="savew2"><rect x="96.0" y="624.9000000000001" width="148" height="50" rx="9" class="bx lane-daily"/><text class="lbl"><tspan x="170" y="649.9000000000001">Save Updated Weights</tspan></text></g>
<g class="fn node" id="nd_logm" data-id="logm"><rect x="96.0" y="710.9000000000001" width="148" height="50" rx="9" class="bx lane-daily"/><text class="lbl"><tspan x="170" y="735.9000000000001">Log + Metrics</tspan></text></g>
<g class="fn node" id="nd_moredays" data-id="moredays"><polygon points="530,704.9000000000001 604.0,735.9000000000001 530,766.9000000000001 456.0,735.9000000000001" class="dia lane-daily"/><text class="lbl"><tspan x="530" y="735.9000000000001">More Days?</tspan></text></g>
<g class="fn node" id="nd_skip" data-id="skip"><rect x="996.0" y="710.9000000000001" width="148" height="50" rx="24" class="se lane-daily"/><text class="lbl"><tspan x="1070" y="735.9000000000001">Stop</tspan></text></g>
<text class="lglabel" x="40" y="848.9000000000001">Legend:</text>
<rect x="118" y="836.9000000000001" width="26" height="15" rx="3" class="bx lane-data"/>
<text class="lgtxt" x="150" y="848.9000000000001">Process</text>
<polygon points="217,835.9000000000001 230,844.9000000000001 217,853.9000000000001 204,844.9000000000001" class="dia"/>
<text class="lgtxt" x="236" y="848.9000000000001">Decision</text>
<rect x="298" y="836.9000000000001" width="26" height="15" rx="7" class="se"/>
<text class="lgtxt" x="330" y="848.9000000000001">Start / end</text>
<line x1="426" y1="844.9000000000001" x2="452" y2="844.9000000000001" class="edge handoff" marker-end="url(#arr)"/>
<text class="lgtxt" x="460" y="848.9000000000001">Stage handoff</text>
</svg>`;
document.getElementById('sysmap').innerHTML=PIPE_SVG;

// canonical node order for the walking animator (main pipeline path)
const FLOW=['start','download','features','split',
  'tune_start','t_tgt','t_mdl','optuna','sample','cv','fit','buildseq','quicktrain','valmse',
  'report','prune','morefolds','avgmse','moretrials','selbest','moremodels','moretgts','writehp',
  'f_tgt','f_mdl','epoch','early','savew','meta',
  'beginday','forecast','verify','agree','seventh','finetune','savew2','logm','moredays','beginday'];

// backend event id -> diagram node id
const NODEMAP={
  download:'download',features:'features',validate:'features',split:'split',
  tune_start:'tune_start',tune_sample:'sample',sample_hp:'sample',
  cv_fold:'cv',fit_scalers:'fit',build_seq:'buildseq',apply_hp:'quicktrain',
  tune_train:'quicktrain',tune_valid:'valmse',tune_report:'report',
  prune:'prune',tune_prune:'prune',tune_record:'morefolds',avg_rmse:'avgmse',
  tune_best:'selbest',best_hp:'selbest',tune_done:'writehp',
  target_loop:'f_tgt',model_loop:'f_mdl',epoch:'epoch',early_stop:'early',
  save_model:'savew',metadata:'meta',
  enter_bt:'beginday',day_loop:'beginday',forecast:'forecast',verify:'verify',
  confident:'agree',retrain_q:'seventh',daily_ft:'finetune',weekly_rt:'retrain',
  save_w:'savew2',metrics:'logm',log:'logm',more_days:'moredays',final:'moredays'};

function eid(f,t){return `e_${f}_${t}`;}

// tuning now animates inside the single combined diagram, so the old tune-map
// helpers are stubs that forward to the main animator.
function setTuneNode(id){ if(typeof setNode==='function') setNode(id,'active'); }
function clearTuneNodes(){ /* no separate tune map anymore */ }

function findEdge(f,t){let id=eid(f,t);if(document.getElementById(id))return id;id=eid(t,f);return document.getElementById(id)?id:null;}
function clearNodes(){document.querySelectorAll('#sysmap .fn').forEach(g=>g.classList.remove('act'));
  document.querySelectorAll('#sysmap .edge').forEach(e=>e.classList.remove('flow'));_mCur=null;_mQ.length=0;if(_mTimer){clearTimeout(_mTimer);_mTimer=null;}}

// --- graph path-walking animator ---------------------------------------------
// The diagram is a real directed graph (ADJ). A backend event names a target
// node; we find the shortest path ALONG THE ACTUAL ARROWS from the current node
// to the target and light each node + connecting edge in turn. This means a
// decision's "No" follows its No-arrow forward, loop-backs follow the loop
// arrow, etc. — the glow never just reverses through boxes.
const ADJ={"start":["download"],"download":["features"],"features":["feat_lag","split"],"feat_lag":["feat_roll"],"feat_roll":["feat_diff"],"feat_diff":["feat_cross"],"feat_cross":["feat_cyc"],"feat_cyc":["feat_anom"],"feat_anom":["features"],"split":["tune_start"],"tune_start":["t_tgt"],"t_tgt":["t_mdl"],"t_mdl":["optuna"],"optuna":["sample"],"sample":["cv"],"cv":["fit"],"fit":["buildseq"],"buildseq":["quicktrain"],"quicktrain":["valmse"],"valmse":["report"],"report":["prune"],"prune":["morefolds","sample"],"morefolds":["avgmse","cv"],"avgmse":["moretrials"],"moretrials":["sample","selbest"],"selbest":["moremodels"],"moremodels":["moretgts","sample"],"moretgts":["writehp","sample"],"writehp":["f_tgt"],"f_tgt":["f_mdl"],"f_mdl":["epoch","f_tgt"],"epoch":["early","f_mdl"],"early":["savew","epoch"],"savew":["meta"],"meta":["beginday"],"beginday":["forecast"],"forecast":["verify"],"verify":["agree"],"agree":["seventh","skipupd"],"seventh":["finetune","retrain"],"retrain":["savew2"],"finetune":["savew2"],"savew2":["logm"],"logm":["moredays"],"moredays":["skip","beginday"],"skipupd":["logm"]};
const STEP_MS=380;
let _mCur=null;          // current diagram node id
let _mQ=[];              // queue of target node ids to walk to
let _mTimer=null;
let _mPath=[];           // remaining nodes to step through toward the head target
function _bfs(from,to){   // shortest path along real arrows; [] if none
  if(from===to)return [to];
  const q=[[from]], seen={}; seen[from]=1;
  while(q.length){
    const p=q.shift(), last=p[p.length-1];
    for(const nx of (ADJ[last]||[])){
      if(seen[nx])continue; seen[nx]=1;
      const np=p.concat(nx);
      if(nx===to)return np;
      q.push(np);
    }
  }
  return [];
}
// loop-back / return edges: these are the curvy arrows that go "backward" in the
// flow (epoch retry, tuning trial loops, daily next-day). Lighting them makes the
// glow trace big curved loops, which looks messy — so we glow the node but DON'T
// flash these specific edges.
const LOOPBACK_EDGES=new Set([
  'e_early_epoch','e_epoch_f_mdl','e_f_mdl_f_tgt',
  'e_moretrials_sample','e_moremodels_sample','e_moretgts_sample','e_prune_sample',
  'e_morefolds_cv','e_moredays_beginday','e_skipupd_logm']);
function _lightNode(id, prev){
  document.querySelectorAll('#sysmap .edge.flow').forEach(e=>e.classList.remove('flow'));
  if(prev){const e=findEdge(prev,id);
    if(e && !LOOPBACK_EDGES.has(e))document.getElementById(e).classList.add('flow');}
  document.querySelectorAll('#sysmap .fn.act').forEach(g=>g.classList.remove('act'));
  const g=document.getElementById('nd_'+id);if(g)g.classList.add('act');
}
function _mPump(){
  if(_mPath.length===0){
    if(_mQ.length===0){_mTimer=null;return;}
    const tgt=_mQ.shift();
    if(_mCur==null){ _mCur=tgt; _lightNode(tgt,null); _mTimer=setTimeout(_mPump,STEP_MS); return; }
    const path=_bfs(_mCur,tgt);
    if(path.length<=1){ _mCur=tgt; _lightNode(tgt,null); _mTimer=setTimeout(_mPump,STEP_MS); return; }
    _mPath=path.slice(1);   // drop current node
  }
  const prev=_mCur, nx=_mPath.shift();
  _mCur=nx; _lightNode(nx,prev);
  _mTimer=setTimeout(_mPump,STEP_MS);
}
function setNode(id,st){const did=NODEMAP[id];if(!did)return;
  // during reload replay, jump straight to the node (no path-walking) so the
  // backlog doesn't draw loop-back arrows / weird closed shapes.
  if(_replaying){ jumpMainTo(did); return; }
  // ignore a repeat of the node we're already on / already heading to (stops blink)
  const tail=_mQ.length?_mQ[_mQ.length-1]:_mCur;
  if(did===tail) return;
  _mQ.push(did);
  if(!_mTimer)_mPump();
}
// jump the walker straight to a node (no walk-through) — used when a live cycle
// starts so it animates from "forecast", not all the way from "download".
function jumpMainTo(did){
  if(!did)return;
  _mQ.length=0; _mPath.length=0; if(_mTimer){clearTimeout(_mTimer);_mTimer=null;}
  _mCur=did; _lightNode(did,null);
}
function queueNode(id){setNode(id,'active');}
function pumpNodes(){}  // legacy no-op
