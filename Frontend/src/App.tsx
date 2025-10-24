import { useState } from 'react'
import Header from './components/Header'
import Inicio from './components/Inicio'
import Funcionamiento from './components/Funcionamiento'
import Pruebas from './components/Pruebas'
import Footer from './components/Footer'

function App() {
  const [activeSection, setActiveSection] = useState('inicio')

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Header activeSection={activeSection} setActiveSection={setActiveSection} />
      
      <main>
        {activeSection === 'inicio' && <Inicio />}
        {activeSection === 'funcionamiento' && <Funcionamiento />}
        {activeSection === 'pruebas' && <Pruebas />}
      </main>

      <Footer />
    </div>
  )
}

export default App