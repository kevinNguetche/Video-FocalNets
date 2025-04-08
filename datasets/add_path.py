import pandas as pd
import os

def add_video_paths(input_csv_path, output_csv_path, base_path):
    """
    Ajoute les chemins complets aux fichiers vidéos listés dans un CSV et vérifie leur mise à jour.

    Parameters:
        input_csv_path (str): Fichier CSV d'entrée.
        output_csv_path (str): Fichier CSV de sortie.
        base_path (str): Chemin de base contenant les vidéos.

    Returns:
        None
    """

    # Vérifier si le chemin de base existe
    if not os.path.exists(base_path):
        print(f"❌ Erreur : Le chemin de base {base_path} n'existe pas.")
        return

    try:
        # Lire le fichier CSV avec le bon séparateur `,`
        df = pd.read_csv(input_csv_path, sep=",", header=None, names=["filename", "label"])
        total_videos = len(df)
        print(f"✅ {total_videos} vidéos chargées depuis {input_csv_path}")

    except Exception as e:
        print(f"❌ Erreur lors de la lecture du fichier CSV : {e}")
        return

    valid_rows = []
    missing_videos = []

    for index, row in df.iterrows():
        video_name = row["filename"].strip()
        label = row["label"]

        # Construire le chemin complet
        full_path = os.path.join(base_path, video_name)

        # Vérifier l'existence du fichier
        if os.path.exists(full_path):
            valid_rows.append([full_path, label])
        else:
            missing_videos.append(video_name)

    # Résumé et vérification finale
    updated_count = len(valid_rows)
    missing_count = len(missing_videos)

    print(f"\n📊 Résumé de la mise à jour :")
    print(f"🔹 Total de vidéos listées dans `{input_csv_path}` : {total_videos}")
    print(f"✅ Vidéos mises à jour avec succès : {updated_count}")
    print(f"⚠️ Vidéos manquantes (non trouvées localement) : {missing_count}")

    if missing_count > 0:
        print(f"❌ Certaines vidéos n'ont pas été mises à jour ! Vérifiez leur présence dans `{base_path}`.")
        # Sauvegarder la liste des vidéos manquantes pour analyse
        with open("missing_videos.log", "w") as log_file:
            for video in missing_videos:
                log_file.write(video + "\n")
        print(f"📄 Liste des vidéos manquantes enregistrée dans `missing_videos.log`.")

    else:
        print(f"✅ Toutes les vidéos ont été correctement mises à jour !")

    # Sauvegarder les chemins valides
    try:
        updated_df = pd.DataFrame(valid_rows, columns=["path", "label"])
        updated_df.to_csv(output_csv_path, sep=",", index=False, header=False)
        print(f"✅ Fichier mis à jour avec les chemins : {output_csv_path}")
    except Exception as e:
        print(f"❌ Erreur lors de l'enregistrement du fichier : {e}")

# Exécution
if __name__ == "__main__":
    input_csv = "old_train_kinetics400.csv"
    output_csv = "train_kinetics400.csv"
    base_path = "/home/ens/Knguetche/kinetics_resized/k400/train"

    add_video_paths(input_csv, output_csv, base_path)

