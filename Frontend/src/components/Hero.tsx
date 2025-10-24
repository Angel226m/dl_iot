import { Shield, Cpu, Cloud } from 'lucide-react';

const Hero = () => {
  const team = ['Garay Torres Angel', 'Figueroa Quiroz Dario', 'Borjas Espinoza Pierr', 'Conca Flores Omar'];

  return (
    <section id="hero" className="pt-32 pb-16 bg-gradient-to-br from-primary-50 to-primary-200">
      <div className="section-container">
        <div className="text-center max-w-4xl mx-auto">
          <div className="flex justify-center gap-6 mb-8">
            <div className="animate-bounce">
              <div className="bg-gradient-to-br from-primary-500 to-primary-600 p-4 rounded-2xl shadow-lg">
                <Shield className="w-12 h-12 text-white" />
              </div>
            </div>
            <div className="animate-bounce" style={{ animationDelay: '0.1s' }}>
              <div className="bg-gradient-to-br from-secondary-500 to-secondary-600 p-4 rounded-2xl shadow-lg">
                <Cpu className="w-12 h-12 text-white" />
              </div>
            </div>
            <div className="animate-bounce" style={{ animationDelay: '0.2s' }}>
              <div className="bg-gradient-to-br from-accent-500 to-accent-600 p-4 rounded-2xl shadow-lg">
                <Cloud className="w-12 h-12 text-white" />
              </div>
            </div>
          </div>
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-heading bg-gradient-to-r from-primary-600 via-primary-700 to-primary-800 bg-clip-text text-transparent mb-6">
            Sistema Inteligente de Detección de Grietas
          </h1>
          <p className="text-xl md:text-2xl text-secondary-700 mb-6">
            basado en <span className="font-bold text-primary-600">IoT</span> y{' '}
            <span className="font-bold text-primary-700">Deep Learning</span>
          </p>
          <div className="inline-block bg-gradient-to-r from-primary-500 to-primary-600 text-white px-6 py-3 rounded-full shadow-lg hover:scale-105 transition-transform duration-300">
            <p className="text-lg font-semibold">🛡️ CrackGuard: Detección Inteligente, Acceso sin Riesgos</p>
          </div>
          <div className="mt-12">
            <h3 className="text-2xl font-heading text-primary-700 mb-6">Integrantes del Equipo</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {team.map((member, index) => (
                <div
                  key={index}
                  className="card hover:shadow-xl transition-all duration-300"
                >
                  <div className="w-20 h-20 bg-gradient-to-br from-primary-400 to-primary-500 rounded-full mx-auto mb-4 flex items-center justify-center shadow-md">
                    <span className="text-3xl font-bold text-white">{member.split(' ')[0][0]}</span>
                  </div>
                  <p className="text-base font-semibold text-secondary-700">{member}</p>
                </div>
              ))}
            </div>
          </div>
          <div className="mt-16 rounded-3xl overflow-hidden shadow-xl hover:shadow-2xl transition-all duration-300">
            <div className="bg-gradient-to-r from-primary-500 to-primary-700 h-64 flex items-center justify-center">
              <div className="text-center text-white">
                <Cloud className="w-28 h-28 mx-auto mb-6 animate-pulse" />
                <p className="text-2xl font-bold mb-2">Monitoreo IoT en Tiempo Real</p>
                <p className="text-lg">Conectividad y análisis inteligente</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;