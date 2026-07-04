#!/bin/bash

API_KEY="4d15e960b59d4dc08efb46d6da0f061f"

curl -X POST "https://api.fish.audio/v1/tts" \
  -H "Authorization: Bearer ${API_KEY}" \
  -H "Content-Type: application/json" \
  -H "model: s2.1-pro-free" \
  -d @- \
  --output dutch_story.mp3 <<'EOF'
{
  "text": "[Speak in Dutch. Female voice. Warm, friendly Dutch neighbor. Natural conversational pacing.] Het is morgen. De zon schijnt in de straat. Een man loopt buiten. Hij ziet een vrouw. Zij is nieuw in de buurt. [vriendelijk] De man zegt: Goedemorgen! [zachte lach] De vrouw lacht. Zij groet de man ook. De man stelt zich voor. [vriendelijk en nieuwsgierig] Mijn naam is Thomas. Hoe heet jij? [warm] De vrouw zegt: Ik ben Sarah. [natuurlijke pauze] Thomas vraagt hoe het gaat. [opgewekt] Sarah zegt dat alles goed gaat. [bewonderend] Kijk eens naar die mooie bloemen in je tuin, zegt Thomas. [korte pauze] Nu loopt Sarah verder. [vriendelijk afscheid nemend] Zij zegt: Tot ziens! [warm en glimlachend] Thomas antwoordt ook met een groet. Tot ziens, Sarah. [verteller, warm] De buren zijn vriendelijk."
}
EOF
