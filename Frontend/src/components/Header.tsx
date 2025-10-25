import { Shield, Menu, X } from 'lucide-react'
import { useState } from 'react'

interface HeaderProps {
  activeSection: string
  setActiveSection: (section: string) => void
}

const Header = ({ activeSection, setActiveSection }: HeaderProps) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)

  const menuItems = [
    { id: 'inicio', label: 'Inicio' },
    { id: 'funcionamiento', label: 'Funcionamiento' },
    { id: 'pruebas', label: 'Prueba el Sistema' },
  ]

  const handleLogoClick = () => {
    setActiveSection('inicio')
    setIsMenuOpen(false)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <header className="fixed top-0 w-full bg-slate-950/95 backdrop-blur-xl border-b border-slate-800 shadow-lg shadow-cyan-500/10 z-50">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex justify-between items-center">
          {/* Logo */}
          <button
            onClick={handleLogoClick}
            className="flex items-center gap-2 group cursor-pointer focus:outline-none"
            aria-label="Volver a la página de inicio"
          >
            <div className="relative">
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl blur-md opacity-75 group-hover:opacity-100 transition duration-300"></div>
              <div className="relative bg-gradient-to-br from-cyan-500 to-blue-600 p-2 rounded-xl shadow-lg shadow-cyan-500/50 group-hover:scale-105 transition duration-300">
                <Shield className="w-6 h-6 sm:w-7 sm:h-7 text-white" />
              </div>
            </div>
            <span className="text-xl sm:text-2xl font-black bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400 bg-clip-text text-transparent">
              CrackGuard
            </span>
          </button>

          {/* Desktop Menu */}
          <div className="hidden md:flex gap-2">
            {menuItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveSection(item.id)}
                className={`relative px-4 sm:px-6 py-2 rounded-xl font-semibold transition-all duration-300 group ${
                  activeSection === item.id
                    ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-500/50'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700 hover:border-cyan-500/50'
                }`}
                aria-current={activeSection === item.id ? 'page' : undefined}
              >
                {activeSection !== item.id && (
                  <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/0 via-cyan-500/10 to-cyan-500/0 transform -skew-x-12 group-hover:translate-x-full transition-transform duration-500"></div>
                )}
                <span className="relative z-10">{item.label}</span>
              </button>
            ))}
          </div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setIsMenuOpen(!isMenuOpen)}
            className="md:hidden bg-slate-800 border border-slate-700 p-2 rounded-xl hover:bg-slate-700 hover:border-cyan-500/50 transition-all duration-300 focus:outline-none"
            aria-label={isMenuOpen ? 'Cerrar menú' : 'Abrir menú'}
          >
            {isMenuOpen ? (
              <X className="w-6 h-6 text-cyan-400" />
            ) : (
              <Menu className="w-6 h-6 text-cyan-400" />
            )}
          </button>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <div className="md:hidden mt-4 space-y-2 animate-in fade-in slide-in-from-top-4 duration-300">
            {menuItems.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  setActiveSection(item.id)
                  setIsMenuOpen(false)
                }}
                className={`w-full px-4 py-3 rounded-xl font-semibold text-left transition-all duration-300 ${
                  activeSection === item.id
                    ? 'bg-gradient-to-r from-cyan-600 to-blue-600 text-white shadow-lg shadow-cyan-500/50'
                    : 'bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700 hover:border-cyan-500/50'
                }`}
                aria-current={activeSection === item.id ? 'page' : undefined}
              >
                {item.label}
              </button>
            ))}
          </div>
        )}
      </nav>
    </header>
  )
}

export default Header