import { Shield } from 'lucide-react'

interface HeaderProps {
  activeSection: string
  setActiveSection: (section: string) => void
}

const Header = ({ activeSection, setActiveSection }: HeaderProps) => {
  const menuItems = [
    { id: 'inicio', label: 'Inicio' },
    { id: 'funcionamiento', label: 'Funcionamiento' },
    { id: 'pruebas', label: 'Prueba el Sistema' },
  ]

  return (
    <header className="fixed top-0 w-full bg-white/95 backdrop-blur-md shadow-lg z-50">
      <nav className="max-w-7xl mx-auto px-6 py-4">
        <div className="flex justify-between items-center">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="bg-gradient-to-br from-blue-500 to-blue-600 p-2 rounded-lg">
              <Shield className="w-8 h-8 text-white" />
            </div>
            <span className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              CrackGuard
            </span>
          </div>

          {/* Menu */}
          <div className="flex gap-3">
            {menuItems.map((item) => (
              <button
                key={item.id}
                onClick={() => setActiveSection(item.id)}
                className={`px-6 py-2.5 rounded-lg font-semibold transition-all duration-300 ${
                  activeSection === item.id
                    ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>
        </div>
      </nav>
    </header>
  )
}

export default Header