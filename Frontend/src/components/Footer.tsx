import { Shield, Github, Mail } from 'lucide-react'

const Footer = () => {
  return (
    <footer className="bg-gradient-to-br from-slate-950 to-slate-900 text-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
          {/* Brand Section */}
          <div>
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-gradient-to-br from-cyan-500 to-blue-600 p-2 rounded-lg">
                <Shield className="w-7 h-7 text-white" />
              </div>
              <span className="text-xl sm:text-2xl font-bold text-white">CrackGuard</span>
            </div>
            <p className="text-slate-400 text-sm sm:text-base">
              Sistema Inteligente de Detección de Grietas basado en IoT y Deep Learning
            </p>
          </div>

          {/* Team Section */}
          <div>
            <h4 className="text-lg font-semibold text-cyan-400 mb-4">Equipo de Desarrollo</h4>
            <ul className="space-y-2 text-slate-400 text-sm sm:text-base">
              {['Garay Torres Angel', 'Serrano Arias Luis', 'Huari Mora Shandel', 'Cama Sanchez Kevin'].map((member, index) => (
                <li key={index} className="hover:text-cyan-400 transition-colors duration-300">
                  {member}
                </li>
              ))}
            </ul>
          </div>

          {/* Project Info & Social Links */}
          <div>
            <h4 className="text-lg font-semibold text-cyan-400 mb-4">Proyecto Académico</h4>
            <p className="text-slate-400 text-sm sm:text-base mb-4">Octubre - Noviembre 2025</p>
            <div className="flex gap-3">
              <a
                href="https://github.com"
                target="_blank"
                rel="noopener noreferrer"
                aria-label="Visitar nuestro repositorio en GitHub"
                className="bg-slate-800/50 p-3 rounded-lg hover:bg-cyan-500/20 transition-all duration-300 hover:scale-110"
              >
                <Github className="w-5 h-5 text-cyan-400" />
              </a>
              <a
                href="mailto:contact@crackguard.com"
                aria-label="Contactarnos por correo electrónico"
                className="bg-slate-800/50 p-3 rounded-lg hover:bg-cyan-500/20 transition-all duration-300 hover:scale-110"
              >
                <Mail className="w-5 h-5 text-cyan-400" />
              </a>
            </div>
          </div>
        </div>

        {/* Copyright */}
        <div className="border-t border-slate-700 mt-8 pt-6 text-center text-slate-400 text-sm">
          <p>© 2025 CrackGuard. Proyecto académico de la Universidad Nacional de Cañete.</p>
        </div>
      </div>
    </footer>
  )
}

export default Footer