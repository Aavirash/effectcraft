//! Stereo downmix of multichannel audio, shared by preview, waveform and export mixdown (they
//! all read footage audio through `MediaPool::audio_samples`).
//!
//! The matrix is the usual ITU-R BS.775 one: front L/R at unity, the centre and every surround
//! or back channel at −3 dB (1/√2) into both sides' own side (centre into both), the low
//! frequency effects channel left out (as ffmpeg's and swresample's default, `lfe_mix_level` 0;
//! not verified against Adobe's documentation, which does not state it). A back-centre channel
//! reaches both sides at −6 dB. The whole matrix is then scaled by one fixed gain, the reciprocal
//! of the largest row sum (ffmpeg's default normalisation), so that full-scale input can never
//! exceed full scale: 5.1 is scaled by 1/(1 + 0.707 + 0.707). Mono stays duplicated at unity and
//! stereo is untouched.
//!
//! Channel order follows the WAVE / SMPTE convention by channel count (what FilmCraft's AC-3, PCM,
//! FLAC and Opus readers deliver): 3 = L R C, 4 = quad L R Ls Rs, 5 = L R C Ls Rs,
//! 6 = L R C LFE Ls Rs, 7 = L R C LFE Cs Ls Rs, 8 = L R C LFE Lb Rb Ls Rs. AAC arrives in either
//! that order or AAC's own (C L R first, LFE last) depending on its channel configuration
//! ([`aac_order`]). More than 8 channels carry no known layout: only the first two (front L/R)
//! are used.

use std::f32::consts::FRAC_1_SQRT_2;

/// A source channel's contribution to the left and right outputs, before normalisation.
type Pair = [f32; 2];

const FL: Pair = [1.0, 0.0];
const FR: Pair = [0.0, 1.0];
const FC: Pair = [FRAC_1_SQRT_2, FRAC_1_SQRT_2];
const LFE: Pair = [0.0, 0.0];
const SL: Pair = [FRAC_1_SQRT_2, 0.0];
const SR: Pair = [0.0, FRAC_1_SQRT_2];
const CS: Pair = [0.5, 0.5];

/// The per-channel `[left, right]` gains for `count` channels in the given order, normalised.
/// `aac` selects AAC's native channel order. Empty for zero channels.
pub fn stereo_gains(count: usize, aac: bool) -> Vec<Pair> {
    let raw: &[Pair] = match (count, aac) {
        (0, _) => &[],
        (1, _) => &[[1.0, 1.0]],
        (2, _) => &[FL, FR],
        (3, false) => &[FL, FR, FC],
        (3, true) => &[FC, FL, FR],
        (4, false) => &[FL, FR, SL, SR],
        (4, true) => &[FC, FL, FR, CS],
        (5, false) => &[FL, FR, FC, SL, SR],
        (5, true) => &[FC, FL, FR, SL, SR],
        (6, false) => &[FL, FR, FC, LFE, SL, SR],
        (6, true) => &[FC, FL, FR, SL, SR, LFE],
        (7, false) => &[FL, FR, FC, LFE, CS, SL, SR],
        (7, true) => &[FC, FL, FR, SL, SR, CS, LFE],
        (8, false) => &[FL, FR, FC, LFE, SL, SR, SL, SR],
        (8, true) => &[FC, FL, FR, SL, SR, SL, SR, LFE],
        _ => &[FL, FR],
    };
    let mut out = raw.to_vec();
    // Channels beyond a known layout (more than 8) are silent.
    out.resize(count, [0.0, 0.0]);
    let peak = out.iter().map(|g| g[0]).sum::<f32>().max(out.iter().map(|g| g[1]).sum::<f32>());
    // Mono is deliberately left at unity on both sides; so is anything already within range.
    if count > 2 && peak > 1.0 {
        for g in &mut out {
            g[0] /= peak;
            g[1] /= peak;
        }
    }
    out
}

/// The order a source's decoded channels arrive in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Order {
    /// WAVE / SMPTE by channel count: PCM, FLAC, AC-3, Opus and the rest.
    #[default]
    Wav,
    /// AAC that FilmCraft reorders to WAVE / SMPTE (configurations 3 to 6, 11 and 12; 4 becomes
    /// L R C Cs) or that ffmpeg wrote in it (a program config element listing the front pair
    /// first: ffmpeg fills a PCE's elements in its own channel order and reads them back so).
    AacWav,
    /// AAC in its element order (C L R … LFE), which FilmCraft keeps: configuration 7 and a PCE
    /// listing the centre first.
    AacElements,
    /// AAC whose configuration the container doesn't show us (only MP4 / MOV headers are read):
    /// as [`Order::AacWav`] up to 7 channels (configurations 3 to 6, 11 and ffmpeg's 6.1 PCE),
    /// as [`Order::AacElements`] above (configuration 7, ffmpeg's 7.1).
    Aac,
}

/// How FilmCraft's decoder hands over the AAC described by the AudioSpecificConfig `asc`.
pub fn aac_order(asc: &[u8]) -> Order {
    let Ok(asc) = filmcraft_aac::AudioSpecificConfig::parse(asc) else { return Order::Aac };
    match (asc.channel_config, asc.pce) {
        (3..=6 | 11 | 12, _) => Order::AacWav,
        (0, Some(pce)) if pce.front.first().is_some_and(|&(cpe, _)| cpe) => Order::AacWav,
        (0, None) => Order::Aac,
        _ => Order::AacElements,
    }
}

/// The gains for `count` channels arriving in `order`.
pub fn decoded_gains(count: usize, order: Order) -> Vec<Pair> {
    let order = match order {
        Order::Aac if (3..=7).contains(&count) => Order::AacWav,
        Order::Aac => Order::AacElements,
        o => o,
    };
    match (order, count) {
        // C L R Cs, reordered to L R C Cs: the same gains, permuted.
        (Order::AacWav, 4) => {
            let g = stereo_gains(4, true);
            [1, 2, 0, 3].iter().filter_map(|&i| g.get(i).copied()).collect()
        }
        (Order::AacElements, _) => stereo_gains(count, true),
        _ => stereo_gains(count, false),
    }
}

/// Downmix planar `channels` arriving in `order` to interleaved stereo in `out` (`out.len() / 2`
/// frames); frames a channel is too short for count as silence, extra frames in `out` stay as
/// they are.
pub fn downmix_into(channels: &[Vec<f32>], order: Order, out: &mut [f32]) {
    mix(channels, &decoded_gains(channels.len(), order), out);
}

fn mix(channels: &[Vec<f32>], gains: &[Pair], out: &mut [f32]) {
    for (ch, g) in channels.iter().zip(gains) {
        if g[0] == 0.0 && g[1] == 0.0 {
            continue;
        }
        for (o, v) in out.as_chunks_mut::<2>().0.iter_mut().zip(ch) {
            o[0] += v * g[0];
            o[1] += v * g[1];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_channel_count_has_a_bounded_defined_matrix() {
        for aac in [false, true] {
            for n in 0..40usize {
                let g = stereo_gains(n, aac);
                assert_eq!(g.len(), n);
                let (l, r) = (g.iter().map(|p| p[0]).sum::<f32>(), g.iter().map(|p| p[1]).sum::<f32>());
                if n > 2 {
                    assert!(l <= 1.0001 && r <= 1.0001, "{n} channels (aac {aac}): {l} {r}");
                    assert!(l > 0.0 && r > 0.0);
                }
            }
        }
    }

    /// AudioSpecificConfigs as ffmpeg 8.1 and FilmCraft's encoder write them (48 kHz AAC-LC).
    const PCE_FFMPEG_61: &str = "118004c848002000c4400d4c61766336322e32382e31303256e500";
    const PCE_FILMCRAFT_61: &str = "118004c845000108840000";

    fn asc(hex: &str) -> Vec<u8> {
        (0..hex.len() / 2).filter_map(|i| hex.get(2 * i..2 * i + 2).and_then(|b| u8::from_str_radix(b, 16).ok())).collect()
    }

    #[test]
    fn aac_order_follows_the_channel_configuration() {
        // AAC-LC at 48 kHz with channel configuration n
        let config = |n: u8| aac_order(&[0x11, 0x80 | n << 3]);
        for n in [3, 4, 5, 6, 11, 12] {
            assert_eq!(config(n), Order::AacWav, "configuration {n}");
        }
        for n in [1, 2, 7] {
            assert_eq!(config(n), Order::AacElements, "configuration {n}");
        }
        assert_eq!(aac_order(&asc(PCE_FFMPEG_61)), Order::AacWav, "ffmpeg's 6.1 PCE lists L R first");
        assert_eq!(aac_order(&asc(PCE_FILMCRAFT_61)), Order::AacElements, "a PCE listing the centre first");
        assert_eq!(aac_order(&[]), Order::Aac);
    }

    #[test]
    fn decoded_aac_gains_follow_the_order() {
        for n in [3, 5, 6, 7, 8] {
            assert_eq!(decoded_gains(n, Order::AacWav), stereo_gains(n, false), "{n} channels");
            assert_eq!(decoded_gains(n, Order::AacElements), stereo_gains(n, true), "{n} channels");
        }
        let g = decoded_gains(4, Order::AacWav);
        assert_eq!(g.len(), 4);
        assert!(g[0][0] > 0.0 && g[0][1] == 0.0 && g[1][1] > 0.0 && g[1][0] == 0.0, "L R first");
        assert!(g[2][0] > 0.0 && g[2][0] == g[2][1], "the centre is third and feeds both sides");
        assert!(g[3][0] > 0.0 && g[3][0] < g[2][0], "the rear centre is last, below the centre");
        assert_eq!(decoded_gains(4, Order::Wav), stereo_gains(4, false), "PCM 4 channels are quad");
        for n in [3, 4, 5, 6, 7] {
            assert_eq!(decoded_gains(n, Order::Aac), decoded_gains(n, Order::AacWav), "{n} channels");
        }
        for n in [1, 2, 8, 9] {
            assert_eq!(decoded_gains(n, Order::Aac), stereo_gains(n, true), "{n} channels");
        }
    }

    #[test]
    fn five_one_matches_the_bs775_matrix() {
        let k = 1.0 / (1.0 + 2.0 * FRAC_1_SQRT_2);
        let g = stereo_gains(6, false);
        assert!((g[0][0] - k).abs() < 1e-6 && g[0][1] == 0.0);
        assert!((g[2][0] - k * FRAC_1_SQRT_2).abs() < 1e-6 && (g[2][1] - k * FRAC_1_SQRT_2).abs() < 1e-6);
        assert_eq!(g[3], [0.0, 0.0], "LFE is left out");
        assert!((g[4][0] - k * FRAC_1_SQRT_2).abs() < 1e-6 && g[4][1] == 0.0);
        assert!((g[5][1] - k * FRAC_1_SQRT_2).abs() < 1e-6 && g[5][0] == 0.0);
        // AAC order: the centre is first, LFE last
        let a = stereo_gains(6, true);
        assert_eq!((a[0], a[5]), (g[2], g[3]));
        assert_eq!((a[1], a[2]), (g[0], g[1]));
    }

    #[test]
    fn mono_and_stereo_are_untouched_and_wide_layouts_use_the_front_pair() {
        assert_eq!(stereo_gains(1, false), vec![[1.0, 1.0]]);
        assert_eq!(stereo_gains(2, true), vec![FL, FR]);
        let g = stereo_gains(12, false);
        assert_eq!((g[0], g[1], g[2]), (FL, FR, [0.0, 0.0]));
    }

    #[test]
    fn ragged_and_empty_input_never_panics() {
        let mut out = vec![0.0; 8];
        downmix_into(&[], Order::Wav, &mut out);
        downmix_into(&[vec![1.0; 100], vec![1.0; 2], vec![], vec![1.0; 3], vec![1.0; 1], vec![1.0]], Order::Aac, &mut out);
        assert!(out.iter().all(|v| v.is_finite()));
        let mut none: [f32; 0] = [];
        downmix_into(&vec![vec![1.0; 3]; 7], Order::Wav, &mut none);
    }
}
