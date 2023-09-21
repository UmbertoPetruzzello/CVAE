# Combined Data Augmentation for HEp-2 Cell Image Classification

Code Repository of the paper "Combined Data Augmentation for HEp-2 Cell Image Classification"

1. HEp-2_Autoencoder è la cartella dedicata al modello di Autoencoder semplice, utilizzato all'inizio della fase di progettazione. 
   I file sono:
   - model.py script che definisce l'architettura modello
   - HEp-2_AE.py script per il training 
2. HEp-2_6VAE è la cartella dedicata al modello Variational Autoencoder, utilizzato nella prima fase in cui abbiamo lavorato con il dataset Spaciment Level. 
   Manca la parte Condizionale.
   I file sono:
   - model.py script che definisce l'architettura modello
   - HEp-2_VAE.py script per il training
   - HEp-2_VAE_Generate.py script per generate le immagini. Per precisare la classe del pattern, l'intensità e il numero di campioni da generare lo script
     viene lanciato con il seguente comando: 
     ```
     python3 HEp-2_VAE_Generate.py --pattern=nomeclassepattern --intensity=intensitàdesiderata --numsamples=numero
     ```
3. HEp-2_CVAE_CL è la cartella dedicata al progetto ufficiale di tesi. Con gli script presenti in questa cartella sono stati eseguiti gli addestramenti e gli
   esperimenti riportati nel progetto di tesi.
   I file sono:
   - model.py script che definisce il modello CVAE sia con la maschera di segmentazione, sia senza maschera di segmentazione
   - dataloader.py script che definisce l'oggetto HEP2Dataset utile per la creazione del DataLoader
   - train.py script per l'addestramento del modello
   - utils.py script con funzioni utili per salvataggio delle immagini
   
   All'interno della directory 5-fold vi sono gli script utilizzati per la fase di sperimentazione.
   - 5-fold_cross_validation.py è lo script che esegue le sperimentazioni. Prima di lanciare lo script modificare la riga di codice 130 dove viene precisato
     il tipo di esperimento da eseguire:
     ```
     experiment = 'Generated Augmented' #None, Undersampling, Traditional Augmentation, Generative Augmentation, Generated Augmented
     ```
   - ResNet.py modello utilizzato per le sperimentazioni
   - dataloader_classifier.py script che definisce l'oggetto HEP2Dataset utile per la creazione del DataLoader
   - utils.py script con funzioni utili 
   
   All'interno della directory images_generation vi sono gli script per la generazione delle immagini con DA tradizionale e con CVAE
