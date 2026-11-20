"""Public-API image fixtures, separate from every historical image receipt.

The ordered float32 banks below are lossless, hash-checked declarations. They
are retained data definitions, not model samples. Atlas serving includes its
DV12 latent perturbation and selected live/average weights; no output noise
is added during evaluation. Conditional hosts explicitly disable independent
row controls, which do not support conditional atoms in the public API.
"""
from __future__ import annotations

from .reproducibility import DEFAULT_SEED

import base64
from copy import deepcopy
import hashlib
import math
import zlib

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from particlegan import GANTrainer, InputNoise, UpdatePolicy, get_recipe, init
from .definition_quality import five_word_metrics


VERSION = "public-api-images-v1"
MASS_TV_MAX = 0.10

# Generated once from retained banks and pinned source declarations. No
# archive, proposal checkout, legacy trainer, or catalog is needed at runtime.
# Frozen ordered data definitions; compression is storage only.
_BANKS = {'C_vs_O2': {'encoded': 'c-lLA#>e(70i2eWW{(Y%Do@CKTG>x2rndjd<<r9d0IOf;g#',
             'modes': 2,
             'ordered_sha256': 'a297e928b5c8c0cc80a0f87162441132e6e28b12f96b6dd01185ecad72e8bd45',
             'source': {'head': '72b312bc9f479b0203839f8227a3ae9d83bcbdda',
                        'path': 'reports/transfer_suite/img_C_vs_O2_transpose_vs_residual16/reproduce_arms.py',
                        'sha256': 'fa8d85381f91a1a7c37fe05064265daabe5e39020d97a2226c7dd547a4556d06'}},
 'L_chirality2': {'encoded': 'c-lLA#>e(70i2eWW)H#xEsu_gm8UkQcX-g!{{RrD^;r',
                  'modes': 2,
                  'ordered_sha256': '7187951661c730eb8d8d17400d78125134d27e11731acc09e37a8d2e9c312ddf',
                  'source': {'head': 'dbec10a11a21d6367834370298bb8a3de4a7d786',
                             'path': 'reports/transfer_suite/img_L_chirality2_transpose_vs_residual16/reproduce_arms.py',
                             'sha256': '68bf238ffb6a35da45bcc30abbd536636b94878a4bb2753af9c2165ca8614278'}},
 'T_junction2': {'encoded': 'c-lLA#>e(70i2eWW{(XME03;@kUg}LCl^!8UF3!zE&UAuz8v*g',
                 'modes': 2,
                 'ordered_sha256': '11b15eb8093fb7a7695007ff8b2896b4820845fedb97444b05778ad6e4b28f28',
                 'source': {'head': '9a1b085e5a928ca7adecd3946aee867fd3fc2f28',
                            'path': 'reports/transfer_suite/img_T_junction2_transpose_vs_residual16/reproduce_arms.py',
                            'sha256': '1f6a852e4a2b43a8f4626c4670052188c05bc94a16e5ad04fc413cd93e2a87e0'}},
 'b_d2': {'encoded': 'c-lLA#>e(75jZU^&7M+uWQ<Qgx$^kTrzR#REC#Fp0STD>r2',
          'modes': 2,
          'ordered_sha256': 'dcb67e87283e93ccd800878f7291a8b89e66240525ca0745527cf8f129a07915',
          'source': {'head': 'eb2cdd4e13841399c67167bbc5d1baf663e8796a',
                     'path': 'reports/transfer_suite/img_b_d2_transpose_vs_residual16/reproduce_arms.py',
                     'sha256': '546f2266f0f310faa666faf950d0e25e1f0b12eb1354ba3593e261e375087c76'}},
 'barcode_quiet_lr2': {'encoded': 'c-lLA#>e(76>wTwnmvq0X46Z4pzu)lHvrys`r!',
                       'modes': 2,
                       'ordered_sha256': 'b85680af6f379441b40bfb23fcfd9dd65ae2730d939b027c3ec63a5e5505f58b',
                       'source': {'head': 'd5075cb3607a42eda9a44c03e0c9ec2e8156258a',
                                  'path': 'reports/transfer_suite/img_barcode_quiet_lr2_transpose_vs_residual16/reproduce_arms.py',
                                  'sha256': 'd35e255db3088e772613f649b5499932ea03bd04e4210843ad5f6b2d49826f09'}},
 'bars4': {'encoded': 'c-muN0D%U32*w45yZlJ=FDc;xQ%f!$1(X5WhCcv#(J#R',
           'modes': 4,
           'ordered_sha256': 'ea1a988a1e5aa5d875603049aebe35afe0da4ff398a3bbc9e12cd6338c3814af',
           'source': {'head': '664ce464e3add5c65d06c8b324c6f9892e644eec',
                      'path': 'benchmarks/transfer_suite/image_tasks.py',
                      'sha256': '4f070d0879cbaaaa076f82b4683cfe74ea9ed8d85480d1922e249444a752ce58'}},
 'bars8': {'encoded': 'c-muNXs~AhV_aak%a0U)j)uc%IAD{f6pwT;8h)eUH*~`f0RGcofd',
           'modes': 8,
           'ordered_sha256': '1b3d7270a740c326f5cb94edf253a99a15a6d09dbe01a98c8735e12e5b3f60f5',
           'source': {'head': '664ce464e3add5c65d06c8b324c6f9892e644eec',
                      'path': 'benchmarks/transfer_suite/image_tasks.py',
                      'sha256': '4f070d0879cbaaaa076f82b4683cfe74ea9ed8d85480d1922e249444a752ce58'}},
 'blobs4': {'encoded': 'c-muNAO<wpLom7W!wrmv$7pyDkMIBh;b{x;',
            'modes': 4,
            'ordered_sha256': 'aa6de7828779b401898fbeb0d794bcb638e600ce24ec3c7bbc3e7d37c6338f40',
            'source': {'head': '664ce464e3add5c65d06c8b324c6f9892e644eec',
                       'path': 'benchmarks/transfer_suite/image_tasks.py',
                       'sha256': '4f070d0879cbaaaa076f82b4683cfe74ea9ed8d85480d1922e249444a752ce58'}},
 'braille_cell_lr2': {'encoded': 'c-lLA#>e(70i2eWW)H%I)EOIZw!@{L7&~ZbJ~c6}@WAChQr(P8j#}m;V*rPp2Ko',
                      'modes': 2,
                      'ordered_sha256': '317de4be0b2e537136e05b0a48eb921b83229118b176d8f6d1b96a4d610977ec',
                      'source': {'head': '54407478b337a904820c35f3304c565019c3a8a4',
                                 'path': 'reports/transfer_suite/img_braille_cell_lr2_transpose_vs_residual16/reproduce_arms.py',
                                 'sha256': 'ed172ac98ee279ba0c8dc1e1514dbd3aaeab09e89431bdb54c2b82a65c9aa877'}},
 'chirp_up_down2': {'encoded': 'c-lLAhR6Qw86R6*I4vy=D9;1M*wpaYgVZ3@VOIlFcNVXG$ZDu%A8x;$CE$1R%*W?GT>irsKKT7hOnBk*BLG3s{xJ',
                    'modes': 2,
                    'ordered_sha256': 'baa7d2d41f222d12af7bedfa29371409bda3acf9525e2ff4d50ff86e4bea6561',
                    'source': {'head': '56f63d371acc01d16a1a8910d36c179617518c96',
                               'path': 'reports/transfer_suite/img_chirp_up_down2_transpose_vs_residual16/reproduce_arms.py',
                               'sha256': 'dfdc5be91c4e98a788ff9f85aaec51a53166c3d279536450abee3812cfa2cb2e'}},
 'colorize_lr2': {'encoded': 'c-muNpbjuLF180@28ISZ2&S+6zypZ85bmV6Jb3^BWpf>2',
                  'modes': 2,
                  'ordered_sha256': 'b6e9138df884254b9e6628f6bd80bf222505768ebc225b9a6a7ace7bc44f640a',
                  'source': {'head': 'ad2dabdb95532fa9cca9fe9c1965e0cdb05c530f',
                             'path': 'reports/transfer_suite/img_colorize_lr2_transpose_vs_residual16/reproduce_arms.py',
                             'sha256': 'ba7d3b98444e1d0b0e8c9c7edcdc8ea44a5b7c7d3ca24d534923a68f4f7e70e2'}},
 'diag_ramp2': {'encoded': 'c-n>0!3}^Q3<c26zzmQ$0XN$-GdKgx;0z3d?$Txa;Mwnir0JuyfQU#ceNQFmRS!k1%?DyvkZ}vftn_HinrCCyV=-np+nD*N#_e2d#xwW!%WH2$&8%fM(3{zuompIW|F#=%#|(L8w&Cx7OdjrSt8V',
                'modes': 2,
                'ordered_sha256': '2a4b36575b92b47567cbaa93901306d2ce41b12e183eea68432d45fcb6cceea2',
                'source': {'head': 'df3a7410bfbaf70d972d21cec5ca508204eb95ad',
                           'path': 'reports/transfer_suite/img_diag_ramp2_transpose_vs_residual16/reproduce_arms.py',
                           'sha256': '034820c81b8449fca7d4571364f1ea3e1258350592e4986906b7a09340f717b8'}},
 'dof_center_edge2': {'encoded': 'c-mXa=WF|{%HQ_i)JR+DT^V2uVuQq%LgheWGAt`?Ri1wbV-Oo82Ga-PPnHU@>r*}s#vnFG3}z-s4@f*M$jcstL2Qs1%szbbgv=-8zNG~Gw{D`J?fjqZwv`>ab|)_`0%H&xBnFcMiHY=v+Lt63fiZ{;5`*aj@r$RL*l$;8w*z4i8zcrZ6Ql<uuCOM}7KA};kQmH9eDZ|MC*(dt{sRCj&Dym',
                      'modes': 2,
                      'ordered_sha256': 'e3ecabba74d00acab6d677149558de32057ca4e91bd00406ffef4caad31244fa',
                      'source': {'head': '5c09a04291b971205f348b81fc5c5fff91a67ad3',
                                 'path': 'reports/transfer_suite/img_dof_center_edge2_transpose_vs_residual16/reproduce_arms.py',
                                 'sha256': '965acb6cf926a3ab9d1ed330055be55a61593c844e094a1a03d91e53eb2bdcaf'}},
 'dots_count23': {'encoded': 'c-lLA#>e(70i2eWW)H&T%2OND+db&!k;)&;{ssW*gabS',
                  'modes': 2,
                  'ordered_sha256': '12cc43ee77b1e3d056d49cc891e715565aa02f472a4644845d20efa35120efea',
                  'source': {'head': 'e7e54a63e0c6a5cb62f4fcab86e2e9239bea3619',
                             'path': 'reports/transfer_suite/img_dots_count23_transpose_vs_residual16/reproduce_arms.py',
                             'sha256': 'ddef6e034a2867183f322f8bd8f1539330734c00a976c3af88fa023d16f729e0'}},
 'fg_bg_invert2': {'encoded': 'c-ref^wqYX3Ru21+a4LyTb^9J&OOnd8kp<)Vq0WPZ+UVt0Nv>IaR',
                   'modes': 2,
                   'ordered_sha256': '2e3b82bdf1c47995220f0c9df6e6c17c57622d577583dd9cf4b4218855c443a8',
                   'source': {'head': 'c8f5e6bbf703f48dadbeb1403bad799ab9568f3c',
                              'path': 'reports/transfer_suite/img_fg_bg_invert2_transpose_vs_residual16/reproduce_arms.py',
                              'sha256': '685cac28c641da7e3065ca7c1b5d61864a5b98f616b741e432f75d7bac4375dc'}},
 'finder_diag2': {'encoded': 'c-l)#OS4ad;<IObY|$}H9G^Ts{j|VoG<I{^APN&&y9EH=b^74',
                  'modes': 2,
                  'ordered_sha256': '71265a9eca6dd8924b34070b04f24f15ec43ab27cfe2847516f2dc57c109e841',
                  'source': {'head': 'f9ae8f512b3274f21e5a3c3911baf44c9c8b722e',
                             'path': 'reports/transfer_suite/img_finder_diag2_transpose_vs_residual16/reproduce_arms.py',
                             'sha256': '6d0d43be0794f53d650db368a19034e813c1d9b26c9719fa17efa60cd5e9c55c'}},
 'hamburger_kebab2': {'encoded': 'c-lLA#>e(7g*YuO%^n*bD*8#aXQ;|!!vOaI{6h',
                      'modes': 2,
                      'ordered_sha256': 'fd5e035b9a34c8e7da76e420eb4b27a331a079c25773e88ec1cf8a40e7ba2af1',
                      'source': {'head': 'c094bac493ddfe73edd5ee5a8adb926eaf060b1d',
                                 'path': 'reports/transfer_suite/img_hamburger_kebab2_transpose_vs_residual16/reproduce_arms.py',
                                 'sha256': '656afd351d0923e1667d5e4369fe17f0a98f2490f373b1b36cbddde7f64e956c'}},
 'intensity2': {'encoded': 'c-muNpbjuL-fV}A=`BCdVAjk?dt^*+d2#^&qe3zv',
                'modes': 2,
                'ordered_sha256': '05489b1025607126a07709640796daa66e6d0a841697fc6160dff46636879512',
                'source': {'head': '664ce464e3add5c65d06c8b324c6f9892e644eec',
                           'path': 'benchmarks/transfer_suite/image_tasks.py',
                           'sha256': '4f070d0879cbaaaa076f82b4683cfe74ea9ed8d85480d1922e249444a752ce58'}},
 'letterbox_pillar2': {'encoded': 'c-lLA#>e(71vo7&&3+*9Sqj~c%O4{}9srhW<iP',
                       'modes': 2,
                       'ordered_sha256': 'a6b2f0cf9efd215b0ae911eeca8db05e471c0b75d24b006ac59558d9b8e44d8a',
                       'source': {'head': '845eb01bf1c7a8c65daf20f9606b325478f493ff',
                                  'path': 'reports/transfer_suite/img_letterbox_pillar2_transpose_vs_residual16/reproduce_arms.py',
                                  'sha256': 'e2fd1fbdd644e309a64d4c7ea3a89d674562b45035c850f7d1fd595427df25f5'}},
 'mask_inpaint2': {'encoded': 'c-muNaIj|}h#44GuUcihdX<(P14DxykZl(ldesglk6n(z9wCn@7aA&xWIm=GP!E!R+;Sj{O)fMvln8$jiy6RnfZPFbs}_iDi?Cl5s~RN#V^xD}2X-|MD1Jd#gV39WRShwI0|3YJcdP',
                   'modes': 2,
                   'ordered_sha256': '35a9367938881bc0514ecaea3063a99907557cda9a84af8d22cf2e1ccaaa4599',
                   'source': {'head': 'c7936fcb6215728808f9f77177642c0ab0821b79',
                              'path': 'reports/transfer_suite/img_mask_inpaint2_transpose_vs_residual16/reproduce_arms.py',
                              'sha256': '25ae51fa701a16c11eb1a85b14610a5a4bc5b90b28984ad547be68e532a5d032'}},
 'moire_beat2': {'encoded': 'c-l)#OS3<F#>X~ogwwd#!(AQ#7TD@Z',
                 'modes': 2,
                 'ordered_sha256': '5326486aaf791918c93320587e66666e54d7f986838c296ddfe2ba887764761e',
                 'source': {'head': 'cacf2a0ff1f9a67b5cc17e6311db9c37b23476d7',
                            'path': 'reports/transfer_suite/img_moire_beat2_transpose_vs_residual16/reproduce_arms.py',
                            'sha256': '9ab751f94639226390514e78c526f32107563174ce1a00998f2c8f9afbb0aa78'}},
 'play_pause2': {'encoded': 'c-lLA#>e(75jZU^%^rja$>UOkERRhMDe~mFk5Y_mJ~q1syF4xo00{B(_W',
                 'modes': 2,
                 'ordered_sha256': 'e6337309844bc68e36703888b4ec07cd97f2a5f7f75c1131c978f524f4d1e1c7',
                 'source': {'head': '1bbf42aa2eabe4409100e0f8fc52fcedccd48fbc',
                            'path': 'reports/transfer_suite/img_play_pause2_transpose_vs_residual16/reproduce_arms.py',
                            'sha256': '07864807281295877ca6118ebe156a1a3dab79c415e8b2dd14f061840a7c12b9'}},
 'pyramid_valley2': {'encoded': 'c-lLA#>e(76mOWYzz&4ZB8kDpx8AgSmH!)zL2P7oAih|!pnXWV78rxrATgL;kRFiumAOG+3}S=Lf!c>po{;&(xQ~$kVAv_p(;kGei67dhW548~G#G=}$m&4+7Bvq0o|b!dAPi!I#9(?sdO+gc-+RFr#0H7M?87Hd$b4elhtGci0#x-O',
                     'modes': 2,
                     'ordered_sha256': 'c3d123cd0d6ce0bd26afbe4edbe42ad961da3b3641580c2951560a4e1e9e3d83',
                     'source': {'head': '33bb9587362dd0a2a99ca1cd20f9b0963fe7f118',
                                'path': 'reports/transfer_suite/img_pyramid_valley2_transpose_vs_residual16/reproduce_arms.py',
                                'sha256': '103c932c14159f74e473ff546c47bf511397fb9a6e8f6752aaec221e1ccf13df'}},
 'radial_wedge2': {'encoded': 'c-lLA#>e*T0OPc@G<$5ASb1z}Xyryi?jHzwLhi$d0f>A0;Q',
                   'modes': 2,
                   'ordered_sha256': 'fea3006706d8c7996bd7a2666e9038a3d6d4eb843c68aac12877b58f4a3de9e5',
                   'source': {'head': 'aedb87b6148feec4d7c3dd8a69f3d6919b932628',
                              'path': 'reports/transfer_suite/img_radial_wedge2_transpose_vs_residual16/reproduce_arms.py',
                              'sha256': '3e78668e0c4366220b38a0abfb7703ea44f2699f86760ce15f3c7314023b16f1'}},
 'ramp_corner2': {'encoded': 'c-kG&zbnLX9LMoRN$ScZgQD(ANh}Ok-XG;kO1i;6Ai66ImQ#|=B$6bV#UDVK41RP2oz1Egm%$B87C#?1eCyli`FZ_#zdz+tDNQaGQ~z>4Wh&Y9v6@LI-|w;1T8*KO!@yRG6X`Aur`PA6G*WF(g`-CN&Rxaq$9~XZ8~MJpo$E{kT`lP?^Axq~^SHU(3Z8V>Ube02-Kf-}w|*SsKWib6CmmMkeu&-avzWi#iGz!oXrBG&@ub7#)?XH4eJt1Psf#?GblBJCV4QwFj+()JUF7kk!|rQS&CkCx%pG@WXfa+qdCQO`Q#KvuzHU0|s<ZAq=`gc7Gpm`+E{`W2W=D2qXLhHHJf3uzUE8_c%hg35Pddz-c{?-c$y**zI;{Nv_J5XN=GO',
                  'modes': 2,
                  'ordered_sha256': '0a76371f3d256b92f040d7723c2ad264c15d7f023f125df719036211019769e4',
                  'source': {'head': '4e80be439da39c6192ddc842dddc1aa99fffc5d6',
                             'path': 'reports/transfer_suite/img_ramp_corner2_transpose_vs_residual16/reproduce_arms.py',
                             'sha256': 'bd0651b3628567833f076570926dc39c4c48db0f54f5e453d1ed56676a322351'}},
 'smile_frown2': {'encoded': 'c-lLA#>e*T0OPc@G<$qvAT~CPt{#^hz5GC|+tAez@;@yx0D2Apcm',
                  'modes': 2,
                  'ordered_sha256': '4810c5641c6909fff2c0e6f9c9b3e9e2d798ec28263cd1722702500ea3212208',
                  'source': {'head': '178e76a87a9587688424a57f2b11db82d6dbeee8',
                             'path': 'reports/transfer_suite/img_smile_frown2_transpose_vs_residual16/reproduce_arms.py',
                             'sha256': '1c0c6ac7cf21c6cdf32ef2f41c1578ccd23632e72c328c44ced7b3d06ed47794'}},
 'soft_ring2': {'encoded': 'c-muNpbj{D=8PRO#wL%AO^Q4|^9LHtni*-2jOi^;E&u=qVmws',
                'modes': 2,
                'ordered_sha256': 'd975708622f5c44285344cea6fbbc0aa87c4fc2e2a44218535429ee06da8f213',
                'source': {'head': 'a256c0aa559312bb91f1032444bff7a04e38dae7',
                           'path': 'reports/transfer_suite/img_soft_ring2_transpose_vs_residual16/reproduce_arms.py',
                           'sha256': '3d5140e7d10b9edeb4732463116e5d402589ee251020c43327820ddbeff5690c'}},
 'sonar_echo_near_far2': {'encoded': 'c-lLA#>e(7g*YuO&7P3Fv9U1@HD~Pb%M&(#Fk*ae2Kf~sf5r~KJZ-}S0A&UTVg',
                          'modes': 2,
                          'ordered_sha256': '2f6a3c6cf200c3ce9c2a7d39f2b89a4efc99fc13e639175152d035445c899343',
                          'source': {'head': '3673b4f88379db86807eb39281ce66661c1f9ff4',
                                     'path': 'reports/transfer_suite/img_sonar_echo_near_far2_transpose_vs_residual16/reproduce_arms.py',
                                     'sha256': 'bea4cc30631308c2c39f869f0692ed10429c5b029f92bee3c7325c1697cc7118'}},
 'sparse_obs2': {'encoded': 'c-muNAPmyd((FN)TzP7Pfp!nQ?Wd)E03wMO5d',
                 'modes': 2,
                 'ordered_sha256': 'f9055b9110af401de03835ce4cc28854e0c7bddf41bb6c0e47ae51380908299c',
                 'source': {'head': '50dd63350e7e58c97aac01abc4186165a66aea3c',
                            'path': 'reports/transfer_suite/img_sparse_obs2_transpose_vs_residual16/reproduce_arms.py',
                            'sha256': '720267fcf25f744a5a685dd08585e9c34b2ff3afc59ceb6f22edc8fcbb86ea36'}},
 'stairs_asc_desc2': {'encoded': 'c-lLA#>e*T0O7Q>G<#g)*yKnRr=~lJF`pLdXdMRtdSC})',
                      'modes': 2,
                      'ordered_sha256': 'b261a37158128440944c72cc4c56ebe5f30dd560a41e06960a4fa689ce0950f9',
                      'source': {'head': '0a9488d599b064a5e20ccce9344ad5a6a0a0a80b',
                                 'path': 'reports/transfer_suite/img_stairs_asc_desc2_transpose_vs_residual16/reproduce_arms.py',
                                 'sha256': 'bf06aeab091d9b69afea7d20fbb3e6f4ad61c193009bc3ec647fa90d88fb8858'}},
 'stripes2': {'encoded': 'c-muNpfzZ)rwr2*5VCXl$pZk}c^Ba',
              'modes': 2,
              'ordered_sha256': '65f588b70f1713ccf7fe87bf9ebf802d053a8f73ee3ec9d7692b7c06f35a2b32',
              'source': {'head': '664ce464e3add5c65d06c8b324c6f9892e644eec',
                         'path': 'benchmarks/transfer_suite/image_tasks.py',
                         'sha256': '4f070d0879cbaaaa076f82b4683cfe74ea9ed8d85480d1922e249444a752ce58'}},
 'swirl_cw2': {'encoded': 'c-m7acd@_8p=@6uBw@e)jIRBcTjBP6wn_H<3G((e%6WDV?GM^<%?h^r@=e*U$vMw%)iYN668~8HT1jvFfOk9Xrij0@H8{j&w~sy5j_aVX-73vAJHdT!_RdV<_Hwt++v!YEuv=&J#co=un|<6D5Bnay7`uJ^(f0EBF7`{(me{QTVi0>TP;3HF4x|R8b`4OE9#F3m&<rP_Spq;amjKNM*>M19ml4p;2|&9&fNrP(x}_B8rqw{VwE^Au3Fuakn;!$+UIO$356~|lKY{$V0q94NUvB~ZtOWEsA$daP6LKFR{}BowLg7m&eiR7AF91PtzN-',
               'modes': 2,
               'ordered_sha256': '54c25854327731a0af8da154b396627ee2ef4c158ca4491a07f1613dbf452a86',
               'source': {'head': '39f2888d9c19b36e4f638d0204af64ba3b6ee5d5',
                          'path': 'reports/transfer_suite/img_swirl_cw2_transpose_vs_residual16/reproduce_arms.py',
                          'sha256': '227c17a19797af0a95f7569d5eec4b03b90c29527bf24121a29fd5507ca0efdd'}},
 'traffic_stack_rg2': {'encoded': 'c-lLA#>e(75Hm0|*dgPyQ1P=!@@Z*l_8?5Iys`0SFdis*LjEHpKhW-{H2kso4*<h~$0+',
                       'modes': 2,
                       'ordered_sha256': '14fb951cec1f06f0092d68d3b2a6b4cfad56c8dbcc51f7065d07ff7724a3c790',
                       'source': {'head': '228899a9678f24b7f6ab4dfe810d07cdeee612ec',
                                  'path': 'reports/transfer_suite/img_traffic_stack_rg2_transpose_vs_residual16/reproduce_arms.py',
                                  'sha256': '3b1728e06f041cd7417bb627944d731cd4acf929811cae913f3278f34015309b'}},
 'vh_bars2': {'encoded': 'c-muNU|>i~OS1=I1}GS5e0%_=mR!t0A)vMW0E8r5K>',
              'modes': 2,
              'ordered_sha256': '162a39a8b91f962551467297d5630d7a30fe6986d9ef2d09b2a2f864cc1c4639',
              'source': {'head': '39a2534ac7b0bafebb182523b4aa435df315f52a',
                         'path': 'reports/transfer_suite/img_vh_bars2_transpose_vs_residual16/reproduce_arms.py',
                         'sha256': 'cdf52a2b830b753d3699bfeed7de61f26da592d1dd9c3b728fdda65f383844e6'}}}

# A bank's pixel question does not inherit the purpose of the last host that
# happens to use it. Restricted hosts retain their own questions below.
_GOALS = {'C_vs_O2': 'Fixed open C versus closed O pixel templates; RMSE is not an independent topology oracle.',
 'L_chirality2': 'Fixed mirrored L templates; not chirality generalization.',
 'T_junction2': 'Fixed T-junction patterns; not occlusion reasoning.',
 'b_d2': 'Fixed mirrored b/d glyph templates; not general OCR or a dedicated chirality score.',
 'barcode_quiet_lr2': 'Two fixed barcode-like templates with opposite quiet-zone placement; not barcode '
                      'validity or decoding.',
 'bars4': 'Recover all four horizontal/vertical bar positions with sharp pixel fidelity and balanced '
          'output mass.',
 'bars8': 'Recover all eight horizontal/vertical bar positions with pixel fidelity and balanced output '
          'mass; denser finite-support diagnostic.',
 'blobs4': 'Recover four localized 2x2 corner patches with correct position, pixel fidelity and balanced '
           'output mass.',
 'braille_cell_lr2': 'Left-heavy versus right-heavy raised-dot templates; tactile glyph asymmetry, not '
                     'Braille decoding.',
 'chirp_up_down2': 'Two fixed rising/falling spectrogram-like traces; not audio synthesis or frequency '
                   'generalization.',
 'colorize_lr2': 'Fixed grayscale left/right intensity patterns; no color channels or conditional '
                 'grayscale-to-color query.',
 'diag_ramp2': 'Fixed diagonal intensity ramps; not arbitrary image-algebra operations.',
 'dof_center_edge2': 'Fixed center/edge focus profiles; not depth estimation or optics reconstruction.',
 'dots_count23': 'Fixed two-dot/three-dot templates; not counting arbitrary objects, positions or '
                 'cardinalities.',
 'fg_bg_invert2': 'Fixed foreground/background intensity inversions; not conditional image inversion.',
 'finder_diag2': 'Two fixed diagonal finder layouts; not QR recognition or error correction.',
 'hamburger_kebab2': 'Two fixed menu-icon layouts; not UI interaction or semantic object recognition.',
 'intensity2': 'Recover both center-patch intensities (0.35 and 0.85) with correct brightness and balanced '
               'output mass.',
 'letterbox_pillar2': 'Fixed letterbox versus pillarbox border placement; not aspect-ratio inference from '
                      'arbitrary images.',
 'mask_inpaint2': 'Fixed templates named mask-inpaint; no observed image or mask enters G, so no conditional '
                  'inpainting is tested.',
 'moire_beat2': 'Two fixed moiré/beat intensity patterns; not recovery of unseen frequencies or phase.',
 'play_pause2': 'Fixed play/pause icon templates; not video dynamics or button behavior.',
 'pyramid_valley2': 'Fixed bright-center versus dark-center radial intensity patterns; not '
                    'shape-from-shading.',
 'radial_wedge2': 'Fixed radial/wedge patterns; not a segmentation or reconstruction task.',
 'ramp_corner2': 'Fixed corner-ramp intensity templates; not a learned coordinate system.',
 'smile_frown2': 'Fixed smile/frown arcs; not emotion classification or facial-image fidelity.',
 'soft_ring2': 'Fixed soft radial/ring templates; not a continuous stochastic shape family.',
 'sonar_echo_near_far2': 'Two fixed near/far echo-location templates; not acoustic propagation or range '
                         'inference.',
 'sparse_obs2': 'Fixed sparse-observation-like templates; no source-domain input or paired correspondence '
                'tests translation.',
 'stairs_asc_desc2': 'Fixed ascending/descending stair templates; not sequence reasoning.',
 'stripes2': 'Recover both centered horizontal and vertical stripes with pixel contrast and balanced '
             'output mass.',
 'swirl_cw2': 'Fixed opposite-handed swirl patterns; not rotation dynamics or optical flow.',
 'traffic_stack_rg2': 'Two fixed grayscale traffic-stack patterns; not red/green color semantics or traffic '
                      'rules.',
 'vh_bars2': 'Fixed vertical/horizontal bars; same orientation-coverage question as the shipped stripes '
             'family.'}

_HOST_GOALS = {
    "develop-img_residual_bars4": "Test nearest-neighbor residual upsampling on all four horizontal/vertical bar positions; require sharp pixel fidelity and balanced spatial coverage.",
    "develop-img_tiny_generator": "Test whether a width2 generator with a one-dimensional latent covers all four sharp bar-position templates with balanced mass; capacity diagnostic.",
    "develop-img_mean_discriminator": "Expose a mean-only critic's inability to distinguish four equal-mass corner-patch locations; measure spatial fidelity and balanced coverage as an information-negative control.",
    "develop-img_uniform_generator": "Expose a spatially uniform generator's inability to render the centered horizontal and vertical stripes; measure spatial fidelity as a representation-negative control.",
    "pr58": "Test each retained architecture's brightness fidelity and balanced mass for center patches at 0.35 and 0.85; this reuses the shipped intensity data law.",
}

# id, legacy ID, pattern, architecture, width, z_dim, budget, RMSE, origin, query
_HOSTS = [
    ('image-develop-img_bars4-residual_upsample16', 'develop-img_bars4', 'bars4', 'residual_upsample', 16, 8, 600, 0.1, 'catalog-arm-1', None),
    ('image-develop-img_blobs4-residual_upsample16', 'develop-img_blobs4', 'blobs4', 'residual_upsample', 16, 8, 600, 0.1, 'catalog-arm-1', None),
    ('image-develop-img_intensity2-residual_upsample16', 'develop-img_intensity2', 'intensity2', 'residual_upsample', 16, 8, 600, 0.06, 'catalog-arm-1', None),
    ('image-develop-img_residual_bars4-residual_upsample16', 'develop-img_residual_bars4', 'bars4', 'residual_upsample', 16, 8, 600, 0.1, 'catalog-arm-1', None),
    ('image-develop-img_stripes2-residual_upsample16', 'develop-img_stripes2', 'stripes2', 'residual_upsample', 16, 8, 600, 0.1, 'catalog-arm-1', None),
    ('image-pr131-transpose12', 'pr131', 'C_vs_O2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr131-residual_upsample16', 'pr131', 'C_vs_O2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr73-transpose12', 'pr73', 'L_chirality2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr73-residual_upsample16', 'pr73', 'L_chirality2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr68-transpose12', 'pr68', 'T_junction2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr68-residual_upsample16', 'pr68', 'T_junction2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr76-transpose12', 'pr76', 'b_d2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr76-residual_upsample16', 'pr76', 'b_d2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr166-transpose12', 'pr166', 'barcode_quiet_lr2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr166-residual_upsample16', 'pr166', 'barcode_quiet_lr2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-develop-img_bars8-transpose12', 'develop-img_bars8', 'bars8', 'transpose', 12, 8, 600, 0.1, 'catalog-arm-1', None),
    ('image-pr170-transpose12', 'pr170', 'braille_cell_lr2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr170-residual_upsample16', 'pr170', 'braille_cell_lr2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr154-transpose12', 'pr154', 'chirp_up_down2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr154-residual_upsample16', 'pr154', 'chirp_up_down2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr62-transpose12', 'pr62', 'diag_ramp2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr62-residual_upsample16', 'pr62', 'diag_ramp2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr71-transpose12', 'pr71', 'dof_center_edge2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr71-residual_upsample16', 'pr71', 'dof_center_edge2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr74-transpose12', 'pr74', 'dots_count23', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr74-residual_upsample16', 'pr74', 'dots_count23', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr66-transpose12', 'pr66', 'fg_bg_invert2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr66-residual_upsample16', 'pr66', 'fg_bg_invert2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr79-transpose12', 'pr79', 'finder_diag2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr79-residual_upsample16', 'pr79', 'finder_diag2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr151-transpose12', 'pr151', 'hamburger_kebab2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr151-residual_upsample16', 'pr151', 'hamburger_kebab2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr58-transpose12', 'pr58', 'intensity2', 'transpose', 12, 8, 600, 0.06, 'catalog-arm-1', None),
    ('image-pr58-residual_upsample16', 'pr58', 'intensity2', 'residual_upsample', 16, 8, 600, 0.06, 'catalog-arm-2', None),
    ('image-pr78-transpose12', 'pr78', 'letterbox_pillar2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr78-residual_upsample16', 'pr78', 'letterbox_pillar2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr150-transpose12', 'pr150', 'moire_beat2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr150-residual_upsample16', 'pr150', 'moire_beat2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr106-transpose12', 'pr106', 'play_pause2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr106-residual_upsample16', 'pr106', 'play_pause2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr77-transpose12', 'pr77', 'pyramid_valley2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr77-residual_upsample16', 'pr77', 'pyramid_valley2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr67-transpose12', 'pr67', 'radial_wedge2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr67-residual_upsample16', 'pr67', 'radial_wedge2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr70-transpose12', 'pr70', 'ramp_corner2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr70-residual_upsample16', 'pr70', 'ramp_corner2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr69-transpose12', 'pr69', 'smile_frown2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr69-residual_upsample16', 'pr69', 'smile_frown2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr59-transpose12', 'pr59', 'soft_ring2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr59-residual_upsample16', 'pr59', 'soft_ring2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr159-transpose12', 'pr159', 'sonar_echo_near_far2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr159-residual_upsample16', 'pr159', 'sonar_echo_near_far2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr75-transpose12', 'pr75', 'stairs_asc_desc2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr75-residual_upsample16', 'pr75', 'stairs_asc_desc2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr72-transpose12', 'pr72', 'swirl_cw2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr72-residual_upsample16', 'pr72', 'swirl_cw2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-develop-img_tiny_generator-transpose2', 'develop-img_tiny_generator', 'bars4', 'transpose', 2, 1, 480, 0.1, 'catalog-arm-1', None),
    ('image-pr80-transpose12', 'pr80', 'traffic_stack_rg2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr80-residual_upsample16', 'pr80', 'traffic_stack_rg2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr64-transpose12', 'pr64', 'vh_bars2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr64-residual_upsample16', 'pr64', 'vh_bars2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr65-transpose12', 'pr65', 'colorize_lr2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr65-residual_upsample16', 'pr65', 'colorize_lr2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-pr63-transpose12', 'pr63', 'mask_inpaint2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr63-residual_upsample16', 'pr63', 'mask_inpaint2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-develop-img_mean_discriminator-mean_discriminator12', 'develop-img_mean_discriminator', 'blobs4', 'mean_discriminator', 12, 8, 480, 0.1, 'catalog-arm-1', None),
    ('image-pr61-transpose12', 'pr61', 'sparse_obs2', 'transpose', 12, 8, 600, 0.05, 'catalog-arm-1', None),
    ('image-pr61-residual_upsample16', 'pr61', 'sparse_obs2', 'residual_upsample', 16, 8, 600, 0.05, 'catalog-arm-2', None),
    ('image-develop-img_uniform_generator-uniform_generator12', 'develop-img_uniform_generator', 'stripes2', 'uniform_generator', 12, 8, 480, 0.1, 'catalog-arm-1', None),
    ('image-develop-img_stripes2-source-transpose12', 'develop-img_stripes2', 'stripes2', 'transpose', 12, 8, 600, 0.1, 'checked-in-source-host', None),
    ('image-develop-img_bars4-source-transpose12', 'develop-img_bars4', 'bars4', 'transpose', 12, 8, 600, 0.1, 'checked-in-source-host', None),
    ('image-develop-img_blobs4-source-transpose12', 'develop-img_blobs4', 'blobs4', 'transpose', 12, 8, 600, 0.1, 'checked-in-source-host', None),
    ('image-develop-img_intensity2-source-transpose12', 'develop-img_intensity2', 'intensity2', 'transpose', 12, 8, 600, 0.06, 'checked-in-source-host', None),
    ('image-pr61-conditional-transpose12', 'pr61', 'sparse_obs2', 'transpose', 12, 8, 600, 0.05, 'retrospective-conditional-variant', 'sparse_completion'),
    ('image-pr61-conditional-residual_upsample16', 'pr61', 'sparse_obs2', 'residual_upsample', 16, 8, 600, 0.05, 'retrospective-conditional-variant', 'sparse_completion'),
    ('image-pr63-conditional-transpose12', 'pr63', 'mask_inpaint2', 'transpose', 12, 8, 600, 0.05, 'retrospective-conditional-variant', 'masked_completion'),
    ('image-pr63-conditional-residual_upsample16', 'pr63', 'mask_inpaint2', 'residual_upsample', 16, 8, 600, 0.05, 'retrospective-conditional-variant', 'masked_completion'),
    ('image-pr65-conditional-transpose12', 'pr65', 'colorize_lr2', 'transpose', 12, 8, 600, 0.05, 'retrospective-conditional-variant', 'rgb_assignment'),
    ('image-pr65-conditional-residual_upsample16', 'pr65', 'colorize_lr2', 'residual_upsample', 16, 8, 600, 0.05, 'retrospective-conditional-variant', 'rgb_assignment'),
]

_QUERY_GOALS = {
    "sparse_completion": "Complete the matching sparse occupancy template from its observed top half; reject the other bottom-half completion even when marginal template mass is correct.",
    "masked_completion": "Preserve observed pixels and recover both equally likely completions for the ambiguous border query, while using the distinctive observed pixel to select the correct completion for each disambiguated query.",
    "rgb_assignment": "Given each fixed grayscale left/right intensity image, generate its specified red or blue RGB assignment; reject grayscale outputs and swapped channel assignments.",
}


def _declarations():
    rows, signatures = [], {}
    for case_id, legacy_id, pattern, architecture, width, z_dim, steps, rmse, origin, query in _HOSTS:
        modes = _BANKS[pattern]["modes"]
        row = dict(id=case_id, legacy_ids=[legacy_id], title=f"{pattern}: {architecture} width {width}",
                   pattern=pattern, architecture=architecture, width=width, z_dim=z_dim,
                   default_steps=steps, batch_size=32, query=query, arm_identity=origin,
                   goal=_QUERY_GOALS[query] if query else _HOST_GOALS.get(legacy_id, _GOALS[pattern]),
                   scope="Finite 8x8 paired queries only; no unseen-image, OCR, topology, DSP or natural-colorization generalization claim." if query else _GOALS[pattern],
                   thresholds=dict(hq_min=.9, modes=modes, quality_rmse=rmse,
                                   min_mode_fraction=.5 / modes, observations=24, minimum_stable_checks=5),
                   source=deepcopy(_BANKS[pattern]["source"]),
                   initialization="Public deterministic_orthogonal_: G seed, D seed+1, prior seed+2",
                   scientific_status="NEW_VARIANT_UNMEASURED")
        if query:
            row["title"] = {"sparse_completion": "Paired sparse occupancy completion", "masked_completion": "Conditional ambiguous masked completion", "rgb_assignment": "Conditional finite RGB assignment"}[query] + f": {architecture} width {width}"
        if architecture == "uniform_generator":
            row["control_reason"] = "Known representation failure: a spatially uniform generator cannot render stripe targets. Quality PASS is not expected."
        elif architecture == "mean_discriminator":
            row["control_reason"] = "Information-negative architecture control: equal-mean patch positions are indistinguishable to this critic. Quality FAIL diagnoses the host rather than the API."
        elif width == 2:
            row["control_reason"] = "Retained capacity stress with width2 and z_dim1, not a required convergence qualification."
        signature = (pattern, architecture, width, z_dim, steps, rmse, query)
        if signature in signatures:
            row["alias_of"] = signatures[signature]
            row["same_execution_definition_as"] = signatures[signature]
            row["alias_scope"] = "Same declared host/data/gates/default seed and API recipe resolution; no second independent training evidence. Changed runtime/recipe/seed/serving fields require a distinct run."
            row["alias_identity_fields"] = ["ordered_template_sha256", "architecture", "width", "z_dim",
                                            "query", "default_steps", "batch_size", "thresholds", "sampling",
                                            "initialization", "resolved_recipe", "seed", "max_steps", "source", "runtime"]
        else:
            signatures[signature] = case_id
        rows.append(row)
    return rows


_DECLARATIONS = _declarations()


def ordered_bank_sha256(bank):
    values = np.asarray(bank, dtype="<f4", order="C")
    return hashlib.sha256(values.tobytes()).hexdigest()


def template_bank(pattern, *, device="cpu"):
    """Return a fresh ordered bank; fail closed on an invalid declaration."""
    try:
        declaration = _BANKS[pattern]
    except KeyError as exc:
        raise ValueError(f"unknown image pattern {pattern!r}") from exc
    raw = zlib.decompress(base64.b85decode(declaration["encoded"]))
    expected_shape = (declaration["modes"], 1, 8, 8)
    if (len(raw) != math.prod(expected_shape) * 4
            or hashlib.sha256(raw).hexdigest() != declaration["ordered_sha256"]):
        raise ValueError(f"ordered template bytes changed for {pattern}")
    values = np.frombuffer(raw, dtype="<f4").copy().reshape(expected_shape)
    if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("template pixels must be finite and in [0, 1]")
    return torch.from_numpy(values).to(device)


def _metadata(declaration):
    row = deepcopy(declaration)
    row.update(kind="image", default_recipe="atlas", eval_samples=1024,
               api_contract_version=VERSION,
               sampling={
                   "prior": "32 uniformly sampled trainable particle rows",
                   "prior_exception": "Finite-template architecture controls retain their original 32-row cloud; this is not a learned-MoG qualification host.",
                   "training_real": "Uniform ordered template IDs, independent N(0,.01^2) pixels, clipped to [0,1]",
                   "evaluation": "Public ServedModel.sample/generate, fixed isolated per-context streams, output_noise=False; DV12 latent perturbation and selected weights retained",
                   "cohort": "New public-API variant; historical results are not imported",
               })
    row["thresholds"].update(distribution_tv_max=MASS_TV_MAX,
                              finite_template_tv_max=MASS_TV_MAX)
    row["ordered_template_sha256"] = _BANKS[row["pattern"]]["ordered_sha256"]
    if row.get("query"):
        counts = {"sparse_completion": [1, 1], "masked_completion": [2, 1, 1], "rgb_assignment": [1, 1]}[row["query"]]
        row["conditional_valid_mode_counts"] = counts
        row["conditional_target_masses"] = [[1 / count] * count for count in counts]
        row["conditional_context_masses"] = [1 / len(counts)] * len(counts)
        row["sampling"]["training_real"] = "Uniform declared contexts; uniform valid completion within context; independent N(0,.01^2) output pixels, clipped to [0,1]"
        row["conditional_api_overrides"] = dict(row_evidence_gate=False,
                                               particle_birth_death=False,
                                               birth_death_backend="knn",
                                               birth_death_feature_scale="none",
                                               birth_death_isolation=False,
                                               birth_death_cells=64)
        row["thresholds"].update(paired_hq_min=.9)
        if row["query"] == "rgb_assignment":
            row["thresholds"]["chromatic_order_accuracy_min"] = .9
        else:
            row["thresholds"]["observed_rmse_max"] = .05
    return row


def list_cases():
    """Enumerate distinct hosts and aliases without constructing a model."""
    return [*[_metadata(row) for row in _DECLARATIONS], _word_metadata()]


def _case(case_id):
    matches = [row for row in _DECLARATIONS if row["id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"unknown image case {case_id!r}")
    return _metadata(matches[0])


def conditional_problem(query, *, device="cpu"):
    """Actual inputs, masks and valid conditional outputs; no label shortcut.

    Sparse occupancy completes the original bank from its observed top half.
    Inpainting includes one ambiguous border-only query and two disambiguated
    queries. Their equally weighted marginal is still the original two modes.
    RGB assignment maps the two original grayscale inputs to explicitly red
    and blue three-channel outputs. It claims only these paired finite inputs.
    """
    if query == "sparse_completion":
        targets = template_bank("sparse_obs2", device=device)
        masks = torch.zeros_like(targets)
        masks[:, :, :4] = 1
        observed = targets * masks
        contexts = torch.cat((observed, masks), dim=1)
        valid = [targets[i:i + 1] for i in range(2)]
    elif query == "masked_completion":
        targets = template_bank("mask_inpaint2", device=device)
        border = torch.ones_like(targets[:1])
        border[:, :, 1:7, 1:7] = 0
        masks = border.expand(3, -1, -1, -1).clone()
        masks[1:, :, 2, 2] = 1
        observed = torch.stack((targets[0], targets[0], targets[1])) * masks
        if not torch.equal(targets[0] * border[0], targets[1] * border[0]):
            raise ValueError("declared ambiguous inpainting border differs")
        if targets[0, 0, 2, 2] == targets[1, 0, 2, 2]:
            raise ValueError("inpainting disambiguating pixel is not distinctive")
        contexts = torch.cat((observed, masks), dim=1)
        valid = [targets, targets[:1], targets[1:]]
    elif query == "rgb_assignment":
        observed = template_bank("colorize_lr2", device=device)
        masks = None
        contexts = observed
        rgb = torch.zeros(2, 3, 8, 8, device=device)
        rgb[0, 0] = observed[0, 0]
        rgb[1, 2] = observed[1, 0]
        targets = rgb
        valid = [targets[:1], targets[1:]]
    else:
        raise ValueError(f"unknown conditional query {query!r}")
    return dict(contexts=contexts, observed=observed, masks=masks,
                valid_targets=valid, marginal_targets=targets)


def _partition(images, targets, quality_rmse, minimum_fraction):
    images = torch.as_tensor(images, dtype=torch.float32, device="cpu").detach()
    targets = torch.as_tensor(targets, dtype=torch.float32, device="cpu").detach()
    if (images.ndim != 4 or targets.ndim != 4 or not len(images) or not len(targets)
            or images.shape[1:] != targets.shape[1:] or images.shape[2:] != (8, 8)
            or images.shape[1] not in (1, 3)
            or not torch.isfinite(images).all() or not torch.isfinite(targets).all()):
        raise ValueError("finite nonempty NCHW 8x8 matching images/targets required")
    if len(targets) > 1:
        separation = (targets[:, None] - targets[None]).square().mean((2, 3, 4)).sqrt()
        separation.fill_diagonal_(float("inf"))
        if float(separation.min()) <= 2 * quality_rmse:
            raise ValueError("template quality neighborhoods overlap")
    distance = (images[:, None] - targets[None]).square().mean((2, 3, 4)).sqrt()
    rmse, assignment = distance.min(1)
    quality = rmse <= quality_rmse
    mass = torch.bincount(assignment, minlength=len(targets)).double() / len(images)
    valid_mass = torch.bincount(assignment[quality], minlength=len(targets)).double() / len(images)
    rejected = 1 - float(valid_mass.sum())
    wanted = torch.full_like(mass, 1 / len(targets))
    return dict(hq=float(quality.double().mean()), mean_rmse=float(rmse.mean()),
                modes=float((valid_mass >= minimum_fraction).sum()),
                distribution_tv=float((mass - wanted).abs().sum() / 2),
                finite_template_tv=float(((valid_mass - wanted).abs().sum() + rejected) / 2),
                rejected_mass=rejected, samples=float(len(images)))


def score_case(case_id, images, *, context_ids=None):
    """Frozen numeric gate; missing contexts and malformed data never vanish."""
    case = _case(case_id)
    bounds = case["thresholds"]
    query = case.get("query")
    failed = []
    if not query:
        metrics = _partition(images, template_bank(case["pattern"]),
                             bounds["quality_rmse"], bounds["min_mode_fraction"])
        if metrics["modes"] < bounds["modes"]:
            failed.append("modes")
        if metrics["hq"] < bounds["hq_min"]:
            failed.append("hq")
        for name in ("distribution_tv", "finite_template_tv"):
            if metrics[name] > bounds[f"{name}_max"]:
                failed.append(name)
    else:
        problem = conditional_problem(query)
        outputs = torch.as_tensor(images, dtype=torch.float32, device="cpu").detach()
        ids = torch.as_tensor(context_ids, device="cpu") if context_ids is not None else None
        count = len(problem["contexts"])
        if (ids is None or ids.ndim != 1 or len(ids) != len(outputs)
                or ids.dtype not in (torch.int32, torch.int64)
                or bool((ids < 0).any()) or bool((ids >= count).any())
                or set(ids.tolist()) != set(range(count))):
            raise ValueError("every declared conditional context needs explicit valid integer IDs")
        marginal = _partition(outputs, problem["marginal_targets"],
                              bounds["quality_rmse"], bounds["min_mode_fraction"])
        partitions = []
        observed_errors, chromatic_accuracies = [], []
        metrics = {f"marginal_{key}": value for key, value in marginal.items()}
        for index, targets in enumerate(problem["valid_targets"]):
            selected = outputs[ids == index]
            partition = _partition(selected, targets, bounds["quality_rmse"], .5 / len(targets))
            partitions.append(partition)
            for key, value in partition.items():
                metrics[f"context_{index}_{key}"] = value
            for name, passes in (
                    ("hq", partition["hq"] >= bounds["paired_hq_min"]),
                    ("modes", partition["modes"] == len(targets)),
                    ("distribution_tv", partition["distribution_tv"] <= bounds["distribution_tv_max"]),
                    ("finite_template_tv", partition["finite_template_tv"] <= bounds["finite_template_tv_max"])):
                if not passes:
                    failed.append(f"context_{index}_{name}")
            if problem["masks"] is not None:
                mask = problem["masks"][index]
                observed = problem["observed"][index]
                per_image = (((selected - observed) * mask).square().sum((1, 2, 3)) / mask.sum()).sqrt()
                error = float(per_image.mean())
                metrics[f"context_{index}_observed_rmse"] = error
                observed_errors.append(error)
                if error > bounds["observed_rmse_max"]:
                    failed.append(f"context_{index}_observed_rmse")
            if query == "rgb_assignment":
                contrast = selected[:, 0].mean((1, 2)) - selected[:, 2].mean((1, 2))
                wanted = 1 if index == 0 else -1
                accuracy = float((wanted * contrast > 0).double().mean())
                metrics[f"context_{index}_chromatic_order_accuracy"] = accuracy
                chromatic_accuracies.append(accuracy)
                if accuracy < bounds["chromatic_order_accuracy_min"]:
                    failed.append(f"context_{index}_chromatic_order_accuracy")
        metrics.update(hq=min(row["hq"] for row in partitions),
                       mean_rmse=max(row["mean_rmse"] for row in partitions),
                       modes=sum(row["modes"] for row in partitions),
                       distribution_tv=max(row["distribution_tv"] for row in partitions),
                       finite_template_tv=max(row["finite_template_tv"] for row in partitions),
                       samples=float(len(outputs)), contexts=float(count))
        if observed_errors:
            metrics["observed_rmse"] = max(observed_errors)
        if chromatic_accuracies:
            metrics["chromatic_order_accuracy"] = min(chromatic_accuracies)
    return dict(metrics=metrics, passed=not failed, failed_bounds=failed)


def oracle_controls(case_id):
    """Deterministic scorer calibration, never training or gate tuning."""
    if case_id == WORD_CASE_ID:
        return word_oracle_controls()
    case = _case(case_id)
    query = case.get("query")
    if not query:
        bank = template_bank(case["pattern"])
        oracle = bank.repeat_interleave(16, dim=0)
        counts = [8 * (len(bank) + 1), *([8] * (len(bank) - 1))]
        unequal = torch.cat([image.unsqueeze(0).expand(count, -1, -1, -1)
                             for image, count in zip(bank, counts)])
        images = dict(oracle=oracle, mass_imbalance=unequal,
                      collapse=bank[:1].expand(len(oracle), -1, -1, -1),
                      global_mean=bank.mean(0, keepdim=True).expand(len(oracle), -1, -1, -1))
        ids = None
    else:
        problem = conditional_problem(query)
        pieces = [bank.repeat_interleave(32 // len(bank), dim=0)
                  for bank in problem["valid_targets"]]
        oracle = torch.cat(pieces)
        ids = torch.arange(len(pieces)).repeat_interleave(32)
        swapped = oracle.clone()
        fixed = [index for index, bank in enumerate(problem["valid_targets"]) if len(bank) == 1]
        for index, other in zip(fixed, reversed(fixed)):
            swapped[index * 32:(index + 1) * 32] = pieces[other]
        images = dict(oracle=oracle, swapped_contexts=swapped,
                      shuffled_correspondence=oracle.roll(16, dims=0),
                      collapse=problem["marginal_targets"][:1].expand(len(oracle), -1, -1, -1))
        if problem["masks"] is not None:
            corrupted = oracle.clone()
            for index in range(len(pieces)):
                mask = problem["masks"][index].bool()
                group = corrupted[index * 32:(index + 1) * 32]
                group[:, mask] = 1 - group[:, mask]
            images["observed_pixel_corruption"] = corrupted
        else:
            gray = problem["observed"][ids].expand(-1, 3, -1, -1).clone()
            images["grayscale_without_color"] = gray
            images["red_blue_channel_swap"] = oracle.flip(1)
    return {name: dict(expected_pass=name == "oracle",
                       **score_case(case_id, values, context_ids=ids))
            for name, values in images.items()}


class ImageGenerator(nn.Module):
    """Original convolutional core with explicit conditional input expansion."""
    def __init__(self, case, *, context_channels=0, output_channels=1):
        super().__init__()
        self.architecture, self.width = case["architecture"], case["width"]
        self.context_channels = context_channels
        self.input = nn.Linear(case["z_dim"] + 64 * context_channels, self.width * 4)
        convolution = nn.Conv2d if self.architecture == "residual_upsample" else nn.ConvTranspose2d
        if self.architecture == "residual_upsample":
            self.first = convolution(self.width, self.width, 3, padding=1)
            self.second = convolution(self.width, self.width, 3, padding=1)
        else:
            self.first = convolution(self.width, self.width, 4, stride=2, padding=1)
            self.second = convolution(self.width, self.width, 4, stride=2, padding=1)
        self.output = nn.Conv2d(self.width, output_channels, 3, padding=1)

    def forward(self, latent, context=None):
        if self.context_channels:
            if context is None or context.shape != (len(latent), self.context_channels, 8, 8):
                raise ValueError("G requires its matching observed image/mask context")
            latent = torch.cat((latent, context.flatten(1)), dim=1)
        elif context is not None:
            raise ValueError("unconditional G does not accept a context")
        value = F.leaky_relu(self.input(latent).reshape(-1, self.width, 2, 2), .2)
        if self.architecture == "residual_upsample":
            value = F.interpolate(value, scale_factor=2, mode="nearest")
            value = value + F.leaky_relu(self.first(value), .2)
            value = F.interpolate(value, scale_factor=2, mode="nearest")
            value = value + F.leaky_relu(self.second(value), .2)
        else:
            value = F.leaky_relu(self.first(value), .2)
            value = F.leaky_relu(self.second(value), .2)
        value = self.output(value).sigmoid()
        if self.architecture == "uniform_generator":
            value = value.mean((2, 3), keepdim=True).expand(-1, -1, 8, 8)
        return value


class ImageCritic(nn.Module):
    def __init__(self, case, *, context_channels=0, output_channels=1):
        super().__init__()
        width = max(12, case["width"])
        self.context_channels = context_channels
        self.mean_only = case["architecture"] == "mean_discriminator"
        self.network = nn.Sequential(
            nn.Conv2d(output_channels + context_channels, width, 3, stride=2, padding=1), nn.LeakyReLU(.2),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.LeakyReLU(.2),
            nn.Flatten(), nn.Linear(8 * width, 1))

    def forward(self, images, context=None):
        if self.mean_only:
            images = images.mean((2, 3), keepdim=True).expand(-1, -1, 8, 8)
        if self.context_channels:
            if context is None or context.shape != (len(images), self.context_channels, 8, 8):
                raise ValueError("D requires the same context for paired real/fake rows")
            images = torch.cat((images, context), dim=1)
        elif context is not None:
            raise ValueError("unconditional D does not accept a context")
        return self.network(images).flatten()


class ImageFixture:
    def __init__(self, case, *, device, seed, recipe_name, max_steps, recipe_overrides=None):
        self.case, self.seed, self.device = case, seed, torch.device(device)
        self.max_steps = case["default_steps"] if max_steps is None else max_steps
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.query = case.get("query")
        self.problem = conditional_problem(self.query, device=device) if self.query else None
        context_channels = 0 if self.problem is None else self.problem["contexts"].shape[1]
        output_channels = 3 if self.query == "rgb_assignment" else 1
        self.centers = template_bank(case["pattern"], device=device)
        overrides = dict(z_dim=case["z_dim"], num_particles=32, prior_kind="particles", sigma_rel=0,
                         batch_size=case["batch_size"])
        if recipe_name not in ("atlas", "e22", "e22_routed"):
            overrides["total_steps"] = case["default_steps"]
        if self.query:
            overrides.update(case["conditional_api_overrides"])
        from .api_contract import validate_recipe_overrides
        self.recipe_overrides = validate_recipe_overrides({**case, "provider": "api_images"}, recipe_name, recipe_overrides)
        overrides.update(self.recipe_overrides)
        self.recipe = get_recipe(recipe_name, **overrides)
        if self.recipe.row_policy != "independent":
            raise ValueError("these convolutional fixtures are not dense routed-bank hosts")
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            self.G = ImageGenerator(case, context_channels=context_channels, output_channels=output_channels).to(device)
            self.D = ImageCritic(case, context_channels=context_channels, output_channels=output_channels).to(device)
            self.prior = self.recipe.make_prior().to(device)
            init.deterministic_orthogonal_(self.G, seed=seed)
            init.deterministic_orthogonal_(self.D, seed=seed + 1)
            init.deterministic_orthogonal_(self.prior, seed=seed + 2)
            if not self.query:
                self.trainer = GANTrainer(self.recipe, self.G, self.D, prior=self.prior, seed=seed,
                                          max_steps=self.max_steps,
                                          model_generator=torch.Generator(device=device).manual_seed(seed + 8))
                self.policy = self.trainer.policy
                self.api_components = ("get_recipe", "Recipe.make_prior", "init.deterministic_orthogonal_", "GANTrainer.step", "GANTrainer.served_model", "ServedModel.sample")
            else:
                self.trainer = None
                self.opt_g, self.opt_d = self.recipe.make_optimizers(self.G, self.D, self.prior,
                                                                   ema_critic=deepcopy(self.D))
                self.loss = self.recipe.make_loss()
                self.spread = self.recipe.make_prior_regularizer(prior=self.prior, reference=self.prior)
                self.penalty = self.recipe.make_critic_penalty(self.opt_d)
                self._context = None
                self.policy = UpdatePolicy(self.recipe, self.G, self.D, prior=self.prior,
                                           generator_optimizer=self.opt_g, critic_optimizer=self.opt_d,
                                           row_semantics="conditional", seed=seed, penalty=self.penalty,
                                           generation=lambda model, latent: model(latent, self._context))
                self.noisy_critic = InputNoise(self.D, generator=self.policy.noise_generator)
                self.api_components = ("get_recipe", "Recipe.make_prior", "init.deterministic_orthogonal_", "Recipe.make_optimizers", "Recipe.make_loss", "Recipe.make_critic_penalty", "UpdatePolicy", "InputNoise", "ServedModel.generate")
        self.data_generator = torch.Generator(device=device).manual_seed(seed + 101)

    @property
    def completed_steps(self):
        return self.policy.completed_steps

    def _real_batch(self):
        batch = self.recipe.batch_size
        if self.problem is None:
            ids = torch.randint(len(self.centers), (batch,), generator=self.data_generator, device=self.device)
            real = self.centers[ids]
            context = None
        else:
            ids = torch.randint(len(self.problem["contexts"]), (batch,), generator=self.data_generator, device=self.device)
            context = self.problem["contexts"][ids]
            real = torch.empty(batch, self.problem["marginal_targets"].shape[1], 8, 8, device=self.device)
            for index, targets in enumerate(self.problem["valid_targets"]):
                positions = (ids == index).nonzero().flatten()
                modes = torch.randint(len(targets), (len(positions),), generator=self.data_generator, device=self.device)
                real[positions] = targets[modes]
        real = (real + .01 * torch.randn(real.shape, generator=self.data_generator, device=self.device)).clamp(0, 1)
        return real, context

    def step(self):
        """One public trainer or matched public conditional-policy update."""
        if self.completed_steps >= self.max_steps:
            raise RuntimeError("image fixture execution budget exhausted")
        real, context = self._real_batch()
        if self.trainer is not None:
            return self.trainer.step(real)
        self._context = context
        flags = [p.requires_grad for p in self.D.parameters()]
        modes = [(module, module.training) for root in (self.G, self.D) for module in root.modules()]
        try:
            noise = self.policy.begin_step(real, execution_limit=self.max_steps)
            self.noisy_critic.std = noise.input_sigma
            self.D.train()
            self.G.eval()
            with torch.no_grad():
                latent, rows = self.prior.sample(len(real), generator=self.policy.latent_generator)
                fake = self.policy.generate(latent, sigma=noise.output_sigma, rows=rows)
            self.policy.observe_critic_pair(real, fake)
            adversarial_d = self.loss.d_loss(self.noisy_critic(real, context), self.noisy_critic(fake, context))
            penalty = self.penalty(self.noisy_critic, real, fake, context)
            loss_d = adversarial_d + penalty
            self.opt_d.zero_grad(set_to_none=True)
            self.policy.before_critic_backward()
            loss_d.backward()
            self.opt_d.step()
            self.policy.after_critic_step()
            self.D.eval().requires_grad_(False)
            self.G.train()
            latent, rows = self.prior.sample(len(real), generator=self.policy.latent_generator)
            generated = self.policy.generate(latent, sigma=noise.output_sigma, rows=rows)
            loss_gan = self.loss.g_loss(self.noisy_critic(generated, context), self.noisy_critic(real, context))
            prior_reg = self.recipe.prior_regularization(self.prior.z, regularizer=self.spread)
            loss_g = loss_gan + prior_reg
            self.opt_g.zero_grad(set_to_none=True)
            self.policy.before_generator_backward()
            loss_g.backward()
            self.policy.after_generator_backward(loss_gan=loss_gan.detach(), loss_critic=adversarial_d.detach())
            self.opt_g.step()
            self.policy.after_generator_step()
            self.policy.finish_step()
            return dict(step=self.completed_steps, loss_g=loss_g.detach(), loss_d=loss_d.detach(),
                        penalty=penalty.detach())
        except Exception:
            self.policy.abort_step()
            raise
        finally:
            for parameter, flag in zip(self.D.parameters(), flags):
                parameter.requires_grad_(flag)
            for module, flag in modes:
                module.training = flag
            self._context = None

    @torch.no_grad()
    def observe(self, n=1024, seed=DEFAULT_SEED + 1000):
        """Score the actual public served law without touching training RNGs."""
        if type(n) is not int or n < 1 or type(seed) is not int or seed < 0:
            raise ValueError("observation size/seed must be positive/nonnegative integers")
        views = []
        if self.problem is None:
            served = self.trainer.served_model()
            stream = torch.Generator(device=self.device).manual_seed(seed)
            samples = served.sample(n, generator=stream, output_noise=False)
            result = score_case(self.case["id"], samples)
            views.append(dict(kind="image", title=self.case["title"], target=self.centers.cpu(),
                              samples=samples.cpu(), caption="Ordered targets and actual public served draws; output noise off, policy latent perturbation retained."))
        else:
            contexts = self.problem["contexts"]
            default_context = contexts[:1].expand(n, -1, -1, -1).detach().clone()
            served = self.policy.served_model(generation_factory=lambda models: lambda model, latent: model(latent, default_context))
            samples, ids = [], []
            for index, context in enumerate(contexts):
                fixed_context = context.unsqueeze(0).expand(n, -1, -1, -1)
                # Capture only frozen inputs; the callback uses its supplied
                # frozen generator, never a live training module.
                served.generation = lambda model, latent, context=fixed_context: model(latent, context)
                stream = torch.Generator(device=self.device).manual_seed(seed + index)
                latent, rows = served.prior.sample(n, generator=stream)
                output = served.generate(latent, generator=stream, output_noise=False, rows=rows)
                samples.append(output)
                ids.append(torch.full((n,), index, dtype=torch.long, device=self.device))
                views.append(dict(kind="image", title=f"Context {index}: desired and generated completion",
                                  target=self.problem["valid_targets"][index].cpu(), samples=output.cpu(),
                                  caption="Same observed input for every draw in this panel; correspondence and mass scored within this context."))
            result = score_case(self.case["id"], torch.cat(samples), context_ids=torch.cat(ids))
            masks = self.problem["masks"]
            views.insert(0, dict(kind="image", title="Actual conditioning inputs" if masks is None else "Observed inputs and masks",
                                 target=self.problem["observed"].cpu(),
                                 samples=self.problem["observed"].cpu() if masks is None else masks.cpu(),
                                 row_labels=["Observed input", "Given input" if masks is None else "Conditioning mask"],
                                 caption="These pixels, and masks where present, enter both G and D; no mode labels enter the networks."))
        result["metrics"].update(completed_steps=float(self.completed_steps),
                                  served_averaged=float(served.source == "averaged"),
                                  output_noise_added=0.,
                                  policy_latent_perturbation=float(served.controller is not None))
        result["views"] = views
        return result

    def state_dict(self):
        """Complete API state plus caller-owned data law/cursor RNG."""
        return dict(version=VERSION, case=deepcopy(self.case), seed=self.seed,
                    recipe=self.recipe.to_dict(), max_steps=self.max_steps,
                    data_generator=self.data_generator.get_state().clone(),
                    api_components=self.api_components,
                    **(self.recipe.prior_regularizer_checkpoint(self.spread) if self.trainer is None else {}),
                    api_state=(self.trainer.state_dict() if self.trainer is not None else self.policy.state_dict()))


def build_case(case_id, *, device="cpu", seed=DEFAULT_SEED, recipe_name="atlas", max_steps=None, recipe_overrides=None):
    if case_id == WORD_CASE_ID:
        from .api_contract import validate_recipe_overrides
        validate_recipe_overrides({**_word_metadata(), "provider": "api_images"}, recipe_name, recipe_overrides)
        return WordFixture(device=device, seed=seed, recipe_name=recipe_name, max_steps=max_steps)
    return ImageFixture(_case(case_id), device=device, seed=seed,
                        recipe_name=recipe_name, max_steps=max_steps, recipe_overrides=recipe_overrides)


WORD_CASE_ID = "image-five-words-joint-ae"
WORDS = ("apple", "grape", "lemon", "melon", "berry")
WORD_CHARS = "abcdefghijklmnopqrstuvwxyz_ "
WORD_LENGTH = 6
WORD_DIM = len(WORD_CHARS) * WORD_LENGTH


def word_bank(*, device="cpu"):
    indices = torch.tensor([[WORD_CHARS.index(c) for c in word + "_"] for word in WORDS],
                           dtype=torch.long, device=device)
    return F.one_hot(indices, num_classes=len(WORD_CHARS)).permute(0, 2, 1).float()


def _word_metadata():
    return dict(id=WORD_CASE_ID, legacy_ids=["source-family-15"], title="Five-word joint autoencoder",
                kind="word", default_recipe="ka2", default_steps=20001, batch_size=256, eval_samples=1024,
                goal="Generate the five equally likely canonical words with confident normalized token probabilities, and reconstruct each of the five matched inputs including underscore padding.",
                scope="Finite vocabulary apple/grape/lemon/melon/berry only. Joint BiGAN inverse reconstruction; no unseen words or natural-language generation. New API-policy variant, not reuse of historical EMA PASS.",
                thresholds=dict(quality_fraction_min=.95, token_probability_min=.90, modes=5,
                                mass_tv_max=.10, minimum_samples=100, reconstruction_exact=True,
                                observations=24, minimum_stable_checks=5),
                sampling=dict(training_words="Independent uniform canonical word IDs",
                              prior="Five uniform trainable 2D particles, original finite-word exception",
                              evaluation="Public ServedModel.generate from its actual prior; selected weights and DV12 latent perturbation retained, output_noise=False; isolated generated/reconstruction streams",
                              reconstruction="Frozen served encoder followed by the same served generator on five correctly paired canonical inputs"),
                recipe_schedule_horizon=20000,
                initialization="Public deterministic_orthogonal_: G seed, D seed+1, E seed+2, prior seed+3",
                api_overrides=dict(num_particles=5, z_dim=2, prior_kind="particles", sigma_rel=0,
                                   batch_size=256, lr=.0006, d_lr_mult=1.5, prior_reg=1.,
                                   betas=(0., .999), ema_decay=.995, network_lr_horizon_cap=None,
                                   input_noise_std=0., output_noise_std=0., output_noise_mode="fixed",
                                   row_evidence_gate=False, particle_birth_death=False,
                                   birth_death_backend="knn", birth_death_feature_scale="none",
                                   birth_death_isolation=False, birth_death_cells=64),
                api_contract_version=VERSION, scientific_status="NEW_VARIANT_UNMEASURED",
                source=dict(head="664ce464e3add5c65d06c8b324c6f9892e644eec", path="examples/five_modes.py",
                            sha256="9da653c131e9889943974e6a340299fb07327b5605182034ba49abb0d2a416b4"),
                evaluator_source=dict(path="benchmarks/toy_audit/definition_quality.py",
                                      sha256="42d835ab7a0937a6c3759dd719691c721663b7857cb990405253e90f7fbd4cc9",
                                      function="five_word_metrics"))


def score_words(probabilities, reconstructions):
    """Full probabilities and paired reconstruction, including padding."""
    measured = five_word_metrics(np.asarray(probabilities), np.asarray(reconstructions))
    failed = []
    for name, passes in (
            ("minimum_samples", measured["sample_count"] >= 100),
            ("quality_fraction", measured["quality_fraction"] >= .95),
            ("modes", measured["modes"] == 5),
            ("mass_tv", measured["mass_tv"] <= .10),
            ("reconstruction_exact", measured["reconstruction_exact"]),
            ("minimum_reconstruction_token_probability", measured["minimum_reconstruction_token_probability"] >= .90)):
        if not passes:
            failed.append(name)
    masses = measured.pop("word_masses")
    expected = measured.pop("passed")
    measured.update({f"mass_{word}": float(mass) for word, mass in zip(WORDS, masses)})
    if expected != (not failed):
        raise RuntimeError("word evaluator/bound declaration differs")
    return dict(metrics=measured, passed=not failed, failed_bounds=failed)


def word_oracle_controls():
    target = word_bank().numpy()
    oracle = np.repeat(target, 40, axis=0)
    diffuse = .5 * oracle + .5 / len(WORD_CHARS)
    swapped_reconstruction = target[[1, 0, 2, 3, 4]]
    wrong_padding = oracle.copy()
    wrong_padding[:, :, -1] = 0
    wrong_padding[:, WORD_CHARS.index(" "), -1] = 1
    unequal = np.repeat(target, [120, 20, 20, 20, 20], axis=0)
    cases = dict(oracle=(oracle, target), diffuse_correct_argmax=(diffuse, target),
                 swapped_reconstruction=(oracle, swapped_reconstruction),
                 collapsed_word=(np.repeat(target[:1], 200, axis=0), target),
                 mass_imbalance=(unequal, target), wrong_padding=(wrong_padding, target))
    return {name: dict(expected_pass=name == "oracle", **score_words(generated, reconstruction))
            for name, (generated, reconstruction) in cases.items()}


class WordGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU(.2), nn.Linear(64, 128),
                                 nn.LeakyReLU(.2), nn.Linear(128, WORD_DIM))

    def forward_logits(self, latent):
        return self.net(latent).reshape(-1, len(WORD_CHARS), WORD_LENGTH)

    def forward(self, latent):
        return self.forward_logits(latent).softmax(1)


class WordEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(WORD_DIM, 128), nn.LeakyReLU(.2),
                                 nn.Linear(128, 64), nn.LeakyReLU(.2), nn.Linear(64, 2))

    def forward(self, words):
        return self.net(words)


class WordJointCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(WORD_DIM + 2, 256), nn.LeakyReLU(.2),
                                 nn.Linear(256, 128), nn.LeakyReLU(.2), nn.Linear(128, 1))

    def forward(self, joint):
        return self.net(joint).flatten()


def _join_words(words, latent):
    return torch.cat((words.flatten(1), latent), dim=1)


class WordFixture:
    """Joint BiGAN word/inverse question through matched public primitives."""
    def __init__(self, *, device, seed, recipe_name, max_steps, components=None):
        self.case, self.device, self.seed = _word_metadata(), torch.device(device), seed
        self.max_steps = self.case["default_steps"] if max_steps is None else max_steps
        if type(self.max_steps) is not int or self.max_steps < 1 or type(seed) is not int or seed < 0:
            raise ValueError("word execution cap/seed must be positive/nonnegative integers")
        overrides = dict(self.case["api_overrides"])
        if recipe_name not in ("atlas", "e22", "e22_routed"):
            overrides["total_steps"] = self.case["recipe_schedule_horizon"]
        self.recipe = get_recipe(recipe_name, **overrides) if components is None else components.recipe
        self.transport = None
        if self.recipe.kinetic_transport_weight or self.recipe.kinetic_transport_local_weight:
            if components is None or components.component_transport != 'output_marginal_v1':
                raise ValueError('joint word transport requires an explicit output_marginal_v1 consumer')
            from particlegan.conditional_transport import OutputMarginalTransport
            self.transport = OutputMarginalTransport(self.recipe)
        if (self.recipe.model != "gan" or self.recipe.conditioning != "scalar"
                or self.recipe.encoder_mode != "none"
                or (self.recipe.num_particles, self.recipe.z_dim, self.recipe.batch_size) != (5, 2, 256)):
            raise ValueError("joint word host requires its five-row, 2D, batch-256 scalar joint objective")
        if self.recipe.row_policy != "independent":
            raise ValueError("joint word host does not implement a dense routed bank")
        devices = [self.device.index if self.device.index is not None else torch.cuda.current_device()] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            if components is None:
                self.G, self.D, self.E = WordGenerator().to(device), WordJointCritic().to(device), WordEncoder().to(device)
                self.prior = self.recipe.make_prior().to(device)
                for offset, module in enumerate((self.G, self.D, self.E, self.prior)):
                    init.deterministic_orthogonal_(module, seed=seed + offset)
            else:
                self.G, self.D, self.E = [components.construct(factory, component=role).to(device)
                    for factory, role in ((WordGenerator, "generator"), (WordJointCritic, "discriminator"),
                                          (WordEncoder, "encoder"))]
                for module, role in ((self.G, "generator"), (self.D, "discriminator"), (self.E, "encoder")):
                    components.initialize(module, component=role)
                self.prior = components.build_prior()
            self.recipe.apply_prior_policy(self.prior)
            # Homogeneous roles are required by the public policy. This splits
            # the old G/E group without changing its optimizer settings.
            self.recipe.apply_prior_policy(self.prior)
            groups = [
                dict(params=list(self.G.parameters()), lr=self.recipe.lr),
                dict(params=list(self.E.parameters()), lr=self.recipe.lr)]
            if self.prior.z.requires_grad:
                groups.append(dict(params=list(self.prior.parameters()), lr=self.recipe.lr * self.recipe.prior_lr_mult,
                    betas=self.recipe.prior_betas if self.recipe.prior_betas is not None else self.recipe.betas))
            self.opt_g = self.recipe.make_generator_optimizer(groups,
                latent_table=self.prior.z if self.prior.z.requires_grad else None)
            self.opt_d = self.recipe.make_critic_optimizer(self.D, ema_critic=deepcopy(self.D))
            self.loss = self.recipe.make_loss()
            self.spread = self.recipe.make_prior_regularizer(prior=self.prior, reference=self.prior)
            self.penalty = self.recipe.make_critic_penalty(self.opt_d, collect_stats=components is not None)
            policy_streams = None if components is None else {
                "latent_generator": components.streams.generator("prior", component="latent", purpose="indices"),
                "penalty_generator": components.streams.generator("noise", component="penalty", purpose="training"),
                "eval_generator": components.streams.generator("eval", component="sampler", purpose="samples"),
                "noise_generator": components.streams.generator("noise", component="generator", purpose="output"),
            }
            self.policy = UpdatePolicy(self.recipe, self.G, self.D, prior=self.prior, encoder=self.E,
                                       generator_optimizer=self.opt_g, critic_optimizer=self.opt_d,
                                       row_semantics="conditional", seed=seed, penalty=self.penalty, streams=policy_streams)
            input_stream = self.policy.noise_generator if components is None else components.streams.generator(
                "noise", component="critic", purpose="input")
            self.noisy_critic = InputNoise(self.D, generator=input_stream)
        self.words = word_bank(device=device)
        self.data_generator = (torch.Generator(device=device).manual_seed(seed + 101) if components is None
            else components.streams.generator("data", component="target", purpose="training"))
        self.api_components = ("get_recipe", "Recipe.make_prior", "init.deterministic_orthogonal_",
                               "Recipe.make_generator_optimizer", "Recipe.make_critic_optimizer",
                               "Recipe.make_loss", "Recipe.make_critic_penalty", "UpdatePolicy",
                               "ServedModel.generate")

    @property
    def completed_steps(self):
        return self.policy.completed_steps

    def step(self):
        if self.completed_steps >= self.max_steps:
            raise RuntimeError("word fixture execution budget exhausted")
        ids = torch.randint(5, (self.recipe.batch_size,), device=self.device, generator=self.data_generator)
        real_words = self.words[ids]
        flags = [parameter.requires_grad for parameter in self.D.parameters()]
        try:
            with torch.no_grad():
                encoded = self.E(real_words)
            joint_real = _join_words(real_words, encoded)
            noise = self.policy.begin_step(joint_real, execution_limit=self.max_steps)
            self.noisy_critic.std = noise.input_sigma
            with torch.no_grad():
                latent, rows = self.prior.sample(len(real_words), generator=self.policy.latent_generator)
                fake_words = self.policy.generate(latent, sigma=noise.output_sigma, rows=rows)
            joint_fake = _join_words(fake_words, latent)
            self.policy.observe_critic_pair(joint_real, joint_fake)
            adversarial_d = self.loss.d_loss(self.noisy_critic(joint_real), self.noisy_critic(joint_fake))
            penalty = self.penalty(self.noisy_critic, joint_real, joint_fake)
            loss_d = adversarial_d + penalty
            self.opt_d.zero_grad(set_to_none=True)
            self.policy.before_critic_backward()
            loss_d.backward()
            self.opt_d.step()
            self.policy.after_critic_step()
            self.D.requires_grad_(False)
            encoded = self.E(real_words)
            latent, rows = self.prior.sample(len(real_words), generator=self.policy.latent_generator)
            fake_words = self.policy.generate(latent, sigma=noise.output_sigma, rows=rows)
            loss_gan = self.loss.joint_g_loss(self.noisy_critic(_join_words(fake_words, latent)),
                                              self.noisy_critic(_join_words(real_words, encoded)))
            loss_g = loss_gan + self.recipe.prior_regularization(self.prior.z, regularizer=self.spread)
            if self.recipe.joint_reconstruction_weight:
                reconstruction = self.recipe.joint_reconstruction_loss(
                    self.G.forward_logits(encoded), real_words, categorical_dim=1)
                loss_g = loss_g + reconstruction
            if self.transport is not None:
                # Categorical probability coordinates only. The joint latent
                # coordinates and inverse encoder retain their original loss.
                loss_g = self.transport.add(loss_g, fake_words.flatten(1), real_words.flatten(1))
            self.opt_g.zero_grad(set_to_none=True)
            self.policy.before_generator_backward()
            if self.recipe.constraint_geometry_mode == 'strict_progress':
                width_noise = (latent - self.prior.z[rows]).detach()
            def protected_evaluator():
                current_latent = self.prior.z[rows] + width_noise
                return (self.loss.joint_g_loss(self.noisy_critic(_join_words(self.G(current_latent), current_latent)),
                    self.noisy_critic(_join_words(real_words, self.E(real_words)))),)
            from particlegan.optim.constraint_geometry import constraint_geometry_backward
            constraint_geometry_backward(loss_g, self.opt_g, (loss_gan,),
                                         protected_evaluator=protected_evaluator)
            self.policy.after_generator_backward(loss_gan=loss_gan.detach(), loss_critic=adversarial_d.detach())
            self.opt_g.step()
            self.policy.after_generator_step()
            self.policy.finish_step()
            result = dict(step=self.completed_steps, loss_d=loss_d.detach(), loss_g=loss_g.detach(), penalty=penalty.detach())
            if self.recipe.joint_reconstruction_weight:
                result["joint_reconstruction"] = reconstruction.detach()
            return result
        except Exception:
            self.policy.abort_step()
            raise
        finally:
            for parameter, flag in zip(self.D.parameters(), flags):
                parameter.requires_grad_(flag)

    @torch.no_grad()
    def observe(self, n=1024, seed=DEFAULT_SEED + 1000, *, generator=None, reconstruction_generator=None):
        if type(n) is not int or n < 1 or type(seed) is not int or seed < 0:
            raise ValueError("observation size/seed must be positive/nonnegative integers")
        served = self.policy.served_model()
        stream = torch.Generator(device=self.device).manual_seed(seed) if generator is None else generator
        latent, rows = served.prior.sample(n, generator=stream)
        generated = served.generate(latent, generator=stream, output_noise=False, rows=rows)
        reconstruction_stream = (torch.Generator(device=self.device).manual_seed(seed + 1)
                                 if reconstruction_generator is None else reconstruction_generator)
        encoded = served.encoder(self.words)
        reconstruction = served.generate(encoded, generator=reconstruction_stream, output_noise=False)
        result = score_words(generated.cpu().numpy(), reconstruction.cpu().numpy())
        result["metrics"].update(completed_steps=float(self.completed_steps), served_averaged=float(served.source == "averaged"),
                                  output_noise_added=0., policy_latent_perturbation=float(served.controller is not None))
        caption = "Rows are letters a–z, underscore and space; columns are six token positions. Target row order: apple_, grape_, lemon_, melon_, berry_. Color intensity is probability, not an argmax display."
        target_labels = [word + "_" for word in WORDS]
        def decoded(probabilities):
            return ["".join(WORD_CHARS[int(token)] for token in row) for row in probabilities.argmax(1).cpu()]
        result["views"] = [dict(kind="text", title="Actual matched word reconstructions",
                                target=self.words.cpu(), samples=reconstruction.cpu(),
                                target_labels=target_labels, sample_labels=decoded(reconstruction),
                                caption="Correct paired words and padding require token probability ≥.90; displayed argmax strings alone do not pass."),
                           dict(kind="image", title="Matched word inputs and actual reconstructions",
                                target=self.words[:, None].cpu(), samples=reconstruction[:, None].cpu(), caption=caption),
                           dict(kind="text", title="Actual generated word strings",
                                target=self.words.cpu(), samples=generated.cpu(), target_labels=target_labels,
                                sample_labels=decoded(generated), caption="The metric also requires high token confidence, all five words and balanced accepted-word mass."),
                           dict(kind="image", title="Canonical vocabulary and actual generated words",
                                target=self.words[:, None].cpu(), samples=generated[:, None].cpu(), caption=caption)]
        return result

    def state_dict(self):
        return dict(version=VERSION, case=deepcopy(self.case), seed=self.seed, recipe=self.recipe.to_dict(),
                    max_steps=self.max_steps, data_generator=self.data_generator.get_state().clone(),
                    **self.recipe.prior_regularizer_checkpoint(self.spread),
                    api_components=self.api_components, api_state=self.policy.state_dict(),
                    **({"component_transport": self.transport.state_dict()} if self.transport is not None else {}))

    def restore_component_transport(self, state):
        """Restore optional consumer counters alongside the existing policy.

        Old inactive fixtures omit this field and require no new mechanism.
        Active fixtures must retain their own consumer evidence on resume.
        """
        if self.transport is None:
            if state is not None:
                raise ValueError('inactive word fixture cannot restore active transport')
        elif state is None:
            raise ValueError('active word fixture is missing transport checkpoint')
        else:
            self.transport.load_state_dict(state)
