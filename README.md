# phoneme_fg_TTS
This project is for phoneme level TTS.

## Update Logs
| operation  | commit id  | 
|---|---|
| reference stride (1 ,2), remove gru  | a6d707fa64027190db |
|  reference stride (1 ,1) | cf9908d  |
|  remove reference conv |   |

## Experiments Reference Attention

|   dataset |   output directory    | reference attention  | decoder attention  | end step | commit id   |
|   ---     |   ---                 |---|---|---|---|
|blz13-19  |113-outdir-7-4-1|FAIL|FAIL| 68k| NONE   |
|full-blz13|113-outdir-7-4-1|FAIL|8000| 32k|  a6d707fa |  
|full-blz13|114-outdir-7-4-2|FAIL| 12k|   | a6d707fa6  |
|full-blz13|113-outdir-7-8-1| - | -  | -  |  cf9908d |  
|full-blz13|113-outdir-7-8-2|  |   |   |  fd1e21d |  
|full-blz13|113-outdir-7-8-3|  |   |   |   |  


