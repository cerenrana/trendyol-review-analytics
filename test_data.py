import pandas as pd
from pathlib import Path

# Test verileri olutur
data = {
    "comment": [
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }rn ok harika, kalitesi mkemmel ve hzl geldi!",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }n geldi, paket hasar grrrrrrm, ok zc",
        "Fiyat olduka uygun, her ey beklediim gibi",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }n ok ktttttt, kalitesi berbat, hi tavsiye etmem",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }rn kusursuz, mkemmel hizmet",
        "Paket aldnda krk buldum, ok hayal krc",
        "Fiyat-performans ok iyi, tekrar satn alrm",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }n materyali beklediim kadar kaliteli deil",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }teri hizmetleri ok yardmc oldu, teekkrler",
        "Kargo irketi ok ge teslimat yapt, hayal krc",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }n renginin fotorafla uyumad iin geri gnnnnnnderdim",
        "ampuan ok etkili, salarm yumuak oldu, ok mutlu",
        "Cilt kremi alerji yapt, ok rahatsz oldum",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }rn ok iyi, rengler canl ve tutma mkkemmel",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }, paralanyor ve salar ktleiyor",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }nyor",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             } formle edilmi",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }zel, uzun sryor",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }ne kremi beyaz izler brakyor, kullanmadm",
        "Krem gzzzzzz evresini yumuatt, krklk azald",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }m ok koku vermiyor, sadece 2 saat tutuyor",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }zel sonu verdi, renk patlyor",
        "Cilt temizleyici derimi kurutuyor, ok sert",
        "Nemlendirici krem ok hafif, hi etkisiz",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }ks kokulu, mucizevi bir rn",
            {                 echo ___BEGIN___COMMAND_OUTPUT_MARKER___;                 PS1="";PS2="";unset HISTFILE;                 EC=$?;                 echo "___BEGIN___COMMAND_DONE_MARKER___$EC";             }tleiyor, rahatsz edici",
        "Lip balm ok nemlendirici, kuru dudaklar iin harika",
        "Toner ok etkili, ciltim parlamay kesti",
        "Serumu uyguladm ksmda dnnnnnnermi olutu",
        "Ya bakm seti harika sonular verdi, ok memnun",
    ]
}

df = pd.DataFrame(data)
output_path = Path("/Users/cerenranaozdemir/trendyol-review-analytics/data/processed/processed_reviews.csv")
df.to_csv(output_path, index=False, encoding="utf-8")
print(f Test verisi oluturuldu: {output_path}")
print(f"Toplam yorum: {len(df)}")
