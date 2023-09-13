/*
Copyright (c) since 2015, Tel Aviv University and Software Modeling Lab

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Tel Aviv University and Software Modeling Lab nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Tel Aviv University and Software Modeling Lab 
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE 
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) 
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
*/

/*
 * generated by Xtext 2.10.0
 */
package tau.smlab.syntech.ui.labeling

import com.google.inject.Inject
import org.eclipse.emf.edit.ui.provider.AdapterFactoryLabelProvider
import org.eclipse.xtext.ui.label.DefaultEObjectLabelProvider
import tau.smlab.syntech.spectra.Define
import tau.smlab.syntech.spectra.LTLAsm
import tau.smlab.syntech.spectra.LTLGar
import tau.smlab.syntech.spectra.Pattern
import tau.smlab.syntech.spectra.Predicate
import tau.smlab.syntech.spectra.TypeDef
import tau.smlab.syntech.spectra.Var
import tau.smlab.syntech.spectra.VarDecl
import tau.smlab.syntech.spectra.VarOwner
import tau.smlab.syntech.spectra.Monitor
import tau.smlab.syntech.spectra.Counter
import tau.smlab.syntech.spectra.DefineRegExp
import tau.smlab.syntech.spectra.EXGar

/**
 * Provides labels for EObjects.
 * 
 * See https://www.eclipse.org/Xtext/documentation/304_ide_concepts.html#label-provider
 */
class SpectraLabelProvider extends DefaultEObjectLabelProvider {

  @Inject
  new(AdapterFactoryLabelProvider delegate) {
    super(delegate);
  }

  // Labels and icons can be computed like this:
  def text(Var ele) {
    return ele.kind.getText + " " + ele.^var.name;
  }

  def text(VarOwner ele) {
    switch (ele) {
      case (SYS):
        return "SYS"
      case (ENV):
        return "ENV"
      case (AUX):
        return "AUX"
    }
    return null;
  }

  def text(VarDecl ele) {
    return ele.name;
  }

  def text(LTLGar ele) {
    var name = ele.name
    if (name === null) {
      name = '';
    }
    return "guarantee " + name;
  }

  def text(LTLAsm ele) {
    var name = ele.name
    if (name === null) {
      name = '';
    }
    return "assumption " + name;
  }

  def text(Define ele) {
    return "define " + ele.defineList.get(0).name;
  }

  def text(TypeDef ele) {
    return "type " + ele.name;
  }

  def text(Predicate ele) {
    return "predicate " + ele.name;
  }

  def text(Pattern ele) {
    return "pattern " + ele.name;
  }

  def text(Monitor ele) {
    return "monitor " + ele.name;
  }

  def text(Counter ele) {
    return "counter " + ele.name;
  }
  
  def text(DefineRegExp ele) {
    return "regexp " + ele.defineRegsList.get(0).name;
  }
  
   def text(EXGar ele) {
    if(ele.name === null) {
    	return "Existential guarantee";
    }
    return "Existential guarantee " + ele.name;
  }

  def image(Var ele) {
    return "package.JPG";
  }

  def image(LTLGar ele) {
    return "LTLGar.gif";
  }
}
