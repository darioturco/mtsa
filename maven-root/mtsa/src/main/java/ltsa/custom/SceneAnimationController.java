package ltsa.custom;

import java.util.BitSet;

import ltsa.lts.Animator;
import ltsa.lts.Relation;
import uk.ac.ic.doc.scenebeans.event.AnimationEvent;
import uk.ac.ic.doc.scenebeans.event.AnimationListener;


public class SceneAnimationController implements 
          Runnable, AnimationMessage, AnimationListener {
    Animator animator;
    OutputActionRegistry  actions;
    ControlActionRegistry controls;
    String [] controlNames;
    volatile BitSet eligible;         // the set of eligible controls
    volatile boolean signalled[];     //set of controls signalled but not cleared
    /* clocking variables */
    Thread ticker;
    boolean debug = false;
    boolean trace = false;
    boolean replayMode = false;
    Object canvas;
    
    public static int LIMIT = 300; //max immediate actions


    public SceneAnimationController(
    	   Animator a, Relation actionMap, Relation controlMap, boolean replay, Object lock) {
        animator = a;
        replayMode = replay;
        canvas = lock;
//        //System.out.println("output registry");
        actions = new OutputActionRegistry(actionMap,this);
//        //System.out.println("control registry");
        controls = new ControlActionRegistry(controlMap,this);
    }

    /* --  output options */

    public void setTrace(boolean b) {trace = b;}
    public void setDebug(boolean b) {debug = b;}

    /*
    * -- actions must be registered before the animation starts
    */

    public void registerAction(String name, AnimationAction action){
        actions.register(name, action);
    }

    /*
    * -- now we can start the controller
    */
    public void start() {
        eligible = animator.initialise(controls.getControls());
        trace = replayMode;
        controlNames = animator.getMenuNames();
        signalled = new boolean[controlNames.length];
        for(int i=0; i<signalled.length; ++i) signalled[i]=true;
        controls.initMap(controlNames);
    }

    public void stop() {
        if (ticker!=null) ticker.interrupt();
    }

    public void restart() {
        if (ticker == null) {
            ticker = new Thread(this);
            ticker.start();
        }
    }
    
    void doReplay() throws InterruptedException {
        while (animator.traceChoice()) {
           eligible = animator.traceStep();
           String todo = animator.actionNameChosen();
           int c = controls.controlled(todo);
           if (c>0) while (!signalled[c]) canvas.wait();  //wait
           actions.doAction(animator.actionNameChosen());
           if (animator.isError()) {
              animator.message("Animation - ERROR state reached");
              return;
           }
        }
        animator.message("Animation - end of Replay");
    }
           
    void doActions() throws InterruptedException {
    	  try {
        while (true) {
        	  doNonControlActions();
        	  if (empty(eligible)) {
                animator.message("Animation - STOP state reached");
                return;
            }
            int choice = -1;
            while((choice = getValidControl())<0) canvas.wait();
            doMenuStep(choice);
        }
        } catch (AnimationException a) {
        	   animator.message("Animation - ERROR state reached "+a);
        }
    }	    


    int getValidControl() {
        for (int i = 0; i<signalled.length; ++i) {
            if (signalled[i] && eligible.get(i)) {
                return i;
            }
        }
        return -1;
    }

		void doMenuStep(int choice) throws AnimationException {
			   eligible=animator.menuStep(choice);
         actions.doAction(animator.actionNameChosen());
         if (animator.isError()) throw new AnimationException();
		}


    void doNonControlActions() throws AnimationException{
        int count = 0;
        while(animator.nonMenuChoice()) {
            eligible = animator.singleStep();
            actions.doAction(animator.actionNameChosen());
            if (animator.isError()) throw new AnimationException();
            ++count;
            if (count>LIMIT)
              throw new AnimationException("immediate action LIMIT exceeded");
        }
    }

 
    private boolean empty(BitSet b) {
        for (int i = 0; i<b.size();++i)
            if (b.get(i)) return false;
        return true;
    }

    public void traceMsg(String msg) {
        if (trace) animator.message(msg);
    }

    public void debugMsg(String msg) {
        if (debug) animator.message(msg);
    }

    public void signalControl(String name) {
    	  synchronized(canvas) {
           if (name.charAt(0)!='~') {
             controls.mapControl(name,signalled,true);
             canvas.notifyAll();
           } else {
             controls.mapControl(name.substring(1),signalled,false);
           }
    	  }
    }

    public void clearControl(String name) {
        controls.mapControl(name,signalled,false);
    }

      public void run () {
        try {
          synchronized(canvas) {
              if (!replayMode) {
                doActions();
              } else {
                doReplay();
              }
          }
        } catch (InterruptedException e){}
        ticker = null;
      }
      
      public void animationEvent( AnimationEvent ev ){
            signalControl(ev.getName());    
      }
      
    public class AnimationException extends Exception{
    public AnimationException() {
        super();
    }
    
    public AnimationException( String msg ) {
        super(msg);
    }
}
}