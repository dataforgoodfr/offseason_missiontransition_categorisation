# offseason_missiontransition_categorisation

## Description

Ce projet a pour but d'automatiser l'affectation de thématiques dans les dispositifs d'aides présents sur la plate-forme Mission Transition Écologique (MTE). Mission Transition Écologique http://mission-transition-ecologique.beta.gouv.fr est un moteur de recherche des aides publiques destinées à la transition écologique pour les entreprises et les acteurs du développement économique.

Les aides et le référentiel des thématiques sont disponibles via l'API de MTE ici: https://mission-transition.beta.gouv.fr/api 

Il semble à priori très difficile avoir des performances suffisantes pour se reposer entièrement sur l'algorithme dans l'affectation des thématiques. En revanche faire un outil de suggestion de thèmes (à choisir parmi plusieurs dizaines) pour la personne en charge de maintenir la base MTE est un objectif plus raisonnable.


## Installation
 
`pip install -r requirements.txt`
`python3 -m spacy download fr_core_news_md`

## Modèle simple

Étant donné le nombre très faible de données annotées (parfois seulement une dizaine d'exemple pour un thème), on part sur un modèle simple, sans recours à du machine learning, basé sur la présence de termes caractéristiques d'un thème ou l'autre. Les paramètres du modèle (à savoir la liste des termes à rechercher pour décider de la présence de tel ou tel thème) ont été ajustés à la main pour 3 thèmes afin de trouver un compromis faux positifs/faux négatifs.

|                       |recall  |precision|
|-----------------------|--------|---------|
|Ressources humaines    |  0.91  |     0.18|
|Secteur bois           |  0.82  |     0.48|
|Mobilité des employés  |  0.74  |     0.42|


Les erreurs du modèle ont été analysées. Il est apparu que beaucoup d'erreurs étaient discutables car les thèmes de l'actuel référentiel V3 sont souvent sujet à interprétation. Quelques erreurs d'annotation ont aussi été détectées. Ces problèmes sont pour l'instant plus limitants que la faible complexité du modèle ou le peu de données. 

Répliquer les résultats:
	`python src/baseline_model.py`
	
Analyse d'erreurs dans le fichier: [report/baseline_error_analysis.org](./report/baseline_error_analysis.org)

On recommande aux experts métier de revoir le référentiel de thèmes et d'établir des règles les plus claires possible afin de laisser la plus petite part possible à l'arbitraire. Il faut ensuite collectivement annoter à nouveau l'ensemble de données.

## Annotation
 
Quelques recommandations pour la redéfinition des thèmes et l'annotation:
  * Labels simples et descriptifs.
  * C'est normal d'allouer un temps conséquent à cette tâche.
  * Écrire un mode d'emploi pour l'annotation pour diminuer la part de l'interprétation. Dans la mesure où les règles sont claires, les non experts métier (bénévoles de D4G par ex.) peuvent contribuer à l'annotation. 
  * Avoir plusieurs personnes pour annoter afin confronter à posteriori les désaccords, réajuster les labels, affiner les règles. 
  * Conserver les annotations des différents annotateurs pour garder la trace des labels problématiques

L'outil gratuit [doccano](https://github.com/doccano/doccano) a été identifié pour faciliter le travail d'annotation. Il est multi-utilisateur et peut tourner sur un cloud (AWS entre autre) et offre une ergonomie qui facilite le travail.

Pour créer un dataset prêt à être importé sur doccano:
`python src/make_doccano_dataset.py`
Voir aussi les données stockées dans `./data/annotation` (redéfinition du référentiel et annotation en cours, bientôt accompagné un mode d'emploi) 

## A suivre
