# AI Music Video Generator

Pipeline pro převod audia a textu písně na hudební video. Projekt podporuje dvě cesty: lokální `image_animation` bez GPU a rozšířený `full_scenes` pipeline s LLM, HF Spaces a volitelným Kaggle backendem.

## Stav implementace

**Funkční a ověřené:** audio analýza přes ffmpeg/numpy, storyboard adapter, SQLite fronta, lokální Ken Burns/loop fallback, image-animation režim, ffmpeg assembly, základní QC a HF klient.

**Doplňovat před produkcí:** R2 adapter, skutečný Kaggle end-to-end běh, Telegram schvalování, autentizovaný dashboard a ověření nasazení na Oracle VM. Externí GPU služby jsou volitelné a bez jejich klíčů se pipeline musí bezpečně vrátit na lokální backend.

## Instalace

```bash
sudo apt-get install -y ffmpeg
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Do `.env` patří pouze rotované klíče. Nikdy necommitujte `.env`, PEM soubory ani exporty tokenů.

## Lokální smoke test

```bash
pytest -q
```

Pro režim s jedním obrázkem:

```bash
python3 -m pipeline.orchestrator \
  --mode image_animation \
  --audio input/song.wav \
  --image character_reference/cover.png \
  --title "Song title"
```

Když HF backend není dostupný nebo nemá kvótu, image animation automaticky použije lokální Ken Burns fallback. Výstup se uloží do `output/`.

## Full scenes

```bash
python3 -m pipeline.orchestrator \
  --audio input/song.wav \
  --title "Song title" \
  --db queue/jobs.db
```

Před spuštěním musí být připraven text `input/song.txt` a příslušné tokeny v `.env`. Externí backendy mají být zapnuté až po samostatném testu healthchecku a kvót.

## Worker

```bash
python3 pipeline/dispatcher.py --loop --sleep 30
```

Worker používá SQLite v režimu WAL a po startu vrací úlohy, které zůstaly ve stavu `running` po pádu procesu, zpět do fronty.

## Nasazení

Unit `deploy/video-agent.service` je šablona pro Oracle VM. Před instalací upravte `WorkingDirectory`, nainstalujte závislosti do systémového Pythonu nebo změňte `ExecStart` na cestu k virtualenv a ověřte oprávnění adresářů. R2, Telegram a Kaggle musí zůstat vypnuté, dokud nejsou jejich credentials a testy hotové.
