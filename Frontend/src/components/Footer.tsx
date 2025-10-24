import { Shield, Github, Mail } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-gradient-to-br from-primary-900 to-primary-800 text-white">
      <div className="section-container">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8 py-8">
          <div>
            <div className="flex items-center gap-2 mb-4">
              <div className="bg-gradient-to-br from-primary-500 to-primary-600 p-2 rounded-lg">
                <Shield className="w-8 h-8 text-white" />
              </div>
              <span className="text-2xl font-heading">CrackGuard</span>
            </div>
            <p className="text-secondary-300">Sistema Inteligente de Detección de Grietas basado en IoT y Deep Learning</p>
          </div>
          <div>
            <h4 className="text-lg font-bold text-primary-300 mb-4">Equipo de Desarrollo</h4>
            <ul className="space-y-2 text-secondary-300">
              <li className="hover:text-primary-400 transition-colors">Garay Torres Angel</li>
              <li className="hover:text-primary-400 transition-colors">Figueroa Quiroz Dario</li>
              <li className="hover:text-primary-400 transition-colors">Borjas Espinoza Pierr</li>
              <li className="hover:text-primary-400 transition-colors">Conca Flores Omar</li>
            </ul>
          </div>
          <div>
            <h4 className="text-lg font-bold text-primary-300 mb-4">Proyecto Académico</h4>
            <p className="text-secondary-300 mb-4">Octubre - Noviembre 2025</p>
            <div className="flex gap-4">
              <a href="#" className="bg-white/10 p-3 rounded-lg hover:bg-white/20 transition-all hover:scale-110"><Github className="w-6 h-6" /></a>
              <a href="#" className="bg-white/10 p-3 rounded-lg hover:bg-white/20 transition-all hover:scale-110"><Mail className="w-6 h-6" /></a>
            </div>
          </div>
        </div>
        <div className="border-t border-primary-700 mt-8 pt-4 text-center text-secondary-400">
          <p>© 2025 CrackGuard. Proyecto académico de la Universidad Nacional de Cañete.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;