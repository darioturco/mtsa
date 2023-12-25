package MTSTools.ac.ic.doc.mtstools.model.operations.DCS.blocking;

public class InstanceAnalyzer {
    public DirectedControllerSynthesisBlocking dcs;
    public boolean knownInstance = false;
    public String instance = "";
    public int n = 0;
    public int k = 0;

    InstanceAnalyzer(DirectedControllerSynthesisBlocking dcs){
        this.dcs = dcs;
    }

    public void setInstance(String instance, int n, int k){
        this.instance = instance;
        this.n = n;
        this.k = k;
        this.knownInstance = true;
    }

    public void printInformation(){
        if(knownInstance){
            // Print instance-n-k
            System.out.println("Instance: " + instance + "-" + n +  "-" + k);
        }
        System.out.println("LTSs: " + dcs.ltss.size());
        System.out.println("Guaranties: " + dcs.guarantees.size());
        System.out.println("Assuptions: " + dcs.assumptions.size());

        // cantidad de ltss
        // cantidad de guaranties
        // cantidad de assumptions
        // cantidad de nodos de la planta
        // cantidad de acciones contolables
        // cantidad de acciones no controlables
        // cantidad de nodos marcados para cada color
        // si la sintesis ya paso (dcs.isFinished())
        // Imprimir la traza de sintesis, metodo de sintesis y cantidad de transiciones y estados expandidos
    }
}
