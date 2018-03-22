"""dictionaries used for switch/case by FeatureExtractor"""

from . import tachibana, knn, neuralnet

single_syl_features_switch_case_dict = {
    'mean spectrum': tachibana.mean_spectrum,
    'mean delta spectrum': tachibana.mean_delta_spectrum,
    'mean cepstrum': tachibana.mean_cepstrum,
    'mean delta cepstrum': tachibana.mean_delta_cepstrum,
    'duration': tachibana.duration,
    'mean spectral centroid': tachibana.mean_spectral_centroid,
    'mean spectral spread': tachibana.mean_spectral_spread,
    'mean spectral skewness': tachibana.mean_spectral_skewness,
    'mean spectral kurtosis': tachibana.mean_spectral_kurtosis,
    'mean spectral flatness': tachibana.mean_spectral_flatness,
    'mean spectral slope': tachibana.mean_spectral_slope,
    'mean pitch': tachibana.mean_pitch,
    'mean pitch goodness': tachibana.mean_pitch_goodness,
    'mean delta spectral centroid': tachibana.mean_delta_spectral_centroid,
    'mean delta spectral spread': tachibana.mean_delta_spectral_spread,
    'mean delta spectral skewness': tachibana.mean_delta_spectral_skewness,
    'mean delta spectral kurtosis': tachibana.mean_delta_spectral_kurtosis,
    'mean delta spectral flatness': tachibana.mean_delta_spectral_flatness,
    'mean delta spectral slope': tachibana.mean_delta_spectral_slope,
    'mean delta pitch': tachibana.mean_delta_pitch,
    'mean delta pitch goodness': tachibana.mean_delta_pitch_goodness,
    'zero crossings': tachibana.zero_crossings,
    'mean amplitude': tachibana.mean_amplitude,
    'mean delta amplitude': tachibana.mean_delta_amplitude,
    'mean smoothed rectified amplitude': knn.mn_amp_smooth_rect,
    'mean RMS amplitude': knn.mn_amp_rms,
    'mean spectral entropy': knn.mean_spect_entropy,
    'mean hi lo ratio': knn.mean_hi_lo_ratio,
    'delta smoothed rectified amplitude': knn.delta_amp_smooth_rect,
    'delta spectral entropy': knn.delta_entropy,
    'delta hi lo ratio': knn.delta_hi_lo_ratio
}

multiple_syl_features_switch_case_dict = {
    'duration group': knn.duration,
    'preceding syllable duration': knn.pre_duration,
    'following syllable duration': knn.foll_duration,
    'preceding silent gap duration': knn.pre_gapdur,
    'following silent gap duration': knn.foll_gapdur
}

neural_net_features_switch_case_dict = {
    'flatwindow': neuralnet.flatwindow,
}
