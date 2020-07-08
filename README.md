# phoneme_fg_TTS
This project is for phoneme level TTS.

## Update Logs
| operation  | commit id  | 
|---|---|
| reference stride (1 ,2), remove gru  | a6d707fa64027190dbedb8a81615e36a4ab90672  |
|  reference stride (1 ,1) |   |
|   |   |

## Experiments Reference Attention

|   dataset |   output directory    | reference attention  | decoder attention  | end step | commit id   |
|   ---     |   ---                 |---|---|---|---|
|blz13-19  |113-outdir-7-4-1|FAIL|FAIL| 68000| NONE   |
|full-blz13|113-outdir-7-4-1|FAIL|8000| 30000|  a6d707fa |  
|full-blz13|114-outdir-7-4-2|FAIL| 12k|   | a6d707fa6  |
|full-blz13|113-outdir-7-8-1|  |   |   |   |  
