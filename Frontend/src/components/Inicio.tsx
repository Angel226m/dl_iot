import { Shield, Cpu, Cloud, AlertTriangle, DollarSign, Clock, Users, TrendingDown, FileX, Zap, Wifi, Brain, MousePointer, ShieldCheck, Target, Calendar, CheckCircle, Layers } from 'lucide-react'

const Inicio = () => {
  const team = [
    'Garay Torres Angel',
    'Figueroa Quiroz Dario',
    'Borjas Espinoza Pierr',
    'Conca Flores Omar'
  ]

  const problems = [
    { icon: <Clock className="w-8 h-8" />, title: 'Inspecciones Manuales Ineficientes', description: 'Procesos lentos que consumen muchos recursos' },
    { icon: <AlertTriangle className="w-8 h-8" />, title: 'Dificultades de Acceso', description: 'Zonas de difícil alcance requieren equipamiento especializado' },
    { icon: <Users className="w-8 h-8" />, title: 'Riesgos Humanos Elevados', description: 'Peligro para inspectores en estructuras altas' },
    { icon: <TrendingDown className="w-8 h-8" />, title: 'Detección Tardía', description: 'Las grietas se detectan cuando ya son peligrosas' },
    { icon: <DollarSign className="w-8 h-8" />, title: 'Costos Altos', description: 'Equipos especializados aumentan costos' },
    { icon: <FileX className="w-8 h-8" />, title: 'Falta de Trazabilidad', description: 'No hay registro digital del historial' },
  ]

  const solutions = [
    { icon: <Zap className="w-10 h-10" />, title: 'Automatización del Monitoreo', description: 'Sistema autónomo que reduce intervención humana' },
    { icon: <Wifi className="w-10 h-10" />, title: 'Acceso Remoto IoT', description: 'Monitoreo desde cualquier ubicación en tiempo real' },
    { icon: <Brain className="w-10 h-10" />, title: 'Análisis con IA', description: 'Deep Learning para detección precisa' },
    { icon: <MousePointer className="w-10 h-10" />, title: 'Activación bajo Demanda', description: 'El usuario decide cuándo inspeccionar' },
    { icon: <ShieldCheck className="w-10 h-10" />, title: 'Mayor Seguridad', description: 'Detección temprana previene accidentes' },
  ]

  return (
    <div className="pt-20">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 py-16">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center max-w-5xl mx-auto">
            <div className="flex justify-center gap-6 mb-8">
              <div className="animate-bounce bg-gradient-to-br from-blue-500 to-blue-600 p-4 rounded-2xl shadow-xl">
                <Shield className="w-12 h-12 text-white" />
              </div>
              <div className="animate-bounce bg-gradient-to-br from-purple-500 to-purple-600 p-4 rounded-2xl shadow-xl" style={{ animationDelay: '0.1s' }}>
                <Cpu className="w-12 h-12 text-white" />
              </div>
              <div className="animate-bounce bg-gradient-to-br from-pink-500 to-pink-600 p-4 rounded-2xl shadow-xl" style={{ animationDelay: '0.2s' }}>
                <Cloud className="w-12 h-12 text-white" />
              </div>
            </div>

            <h1 className="text-5xl md:text-6xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 bg-clip-text text-transparent mb-6">
              Sistema Inteligente de Detección de Grietas en Edificaciones
            </h1>
            
            <p className="text-2xl text-gray-700 mb-6">
              basado en <span className="text-blue-600 font-bold">IoT</span> y{' '}
              <span className="text-purple-600 font-bold">Deep Learning</span>
            </p>

            <div className="inline-block bg-gradient-to-r from-blue-600 to-purple-600 text-white px-8 py-4 rounded-full shadow-2xl mb-12">
              <p className="text-xl font-semibold">🛡️ CrackGuard: Detección Inteligente, Acceso sin Riesgos</p>
            </div>

            {/* Integrantes */}
            <div className="mt-12">
              <h3 className="text-2xl font-bold text-gray-800 mb-6">Integrantes del Equipo</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {team.map((member, index) => (
                  <div key={index} className="bg-white rounded-xl shadow-lg p-6 hover:shadow-2xl transition-all duration-300">
                    <div className="w-20 h-20 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full mx-auto mb-4 flex items-center justify-center shadow-lg">
                      <span className="text-3xl font-bold text-white">{member.split(' ')[0][0]}</span>
                    </div>
                    <p className="text-base font-semibold text-gray-800">{member}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Problemática */}
      <section className="bg-gradient-to-br from-orange-50 to-red-50 py-16">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-gray-800 mb-4 text-center">⚠️ Problemática</h2>
          <p className="text-xl text-gray-600 text-center max-w-3xl mx-auto mb-12">
            Los métodos tradicionales presentan múltiples desafíos que afectan la seguridad
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {problems.map((problem, index) => (
              <div key={index} className="bg-white rounded-xl shadow-lg p-6 hover:shadow-2xl transition-all duration-300">
                <div className="bg-gradient-to-br from-red-400 to-orange-500 text-white p-4 rounded-xl mb-4 flex items-center justify-center shadow-lg">
                  {problem.icon}
                </div>
                <h3 className="text-xl font-bold text-gray-800 mb-3 text-center">{problem.title}</h3>
                <p className="text-gray-600 text-center">{problem.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Solución */}
      <section className="bg-gradient-to-br from-green-50 via-blue-50 to-purple-50 py-16">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-gray-800 mb-4 text-center">✅ Solución Propuesta</h2>
          <p className="text-xl text-gray-600 text-center max-w-3xl mx-auto mb-12">
            CrackGuard integra IoT y Deep Learning para revolucionar la inspección
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {solutions.map((solution, index) => (
              <div key={index} className="bg-white rounded-xl shadow-lg p-6 hover:shadow-2xl transition-all duration-300">
                <div className="bg-gradient-to-br from-green-400 to-blue-500 text-white p-4 rounded-xl mb-4 flex items-center justify-center shadow-lg">
                  {solution.icon}
                </div>
                <h3 className="text-xl font-bold text-gray-800 mb-3 text-center">{solution.title}</h3>
                <p className="text-gray-600 text-center">{solution.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Alcance */}
      <section className="bg-white py-16">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-gray-800 mb-4 text-center">🎯 Alcance del Proyecto</h2>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-12">
            <div className="space-y-6">
              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-start gap-4">
                  <Target className="w-8 h-8 text-blue-600 mt-1" />
                  <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">Desarrollo del MVP</h3>
                    <p className="text-gray-600">Enfoque en detección de grietas con un Producto Mínimo Viable</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-start gap-4">
                  <Layers className="w-8 h-8 text-blue-600 mt-1" />
                  <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">Flujo Completo</h3>
                    <p className="text-gray-600">Captura → Análisis con IA → Visualización de resultados</p>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl shadow-lg p-6">
                <div className="flex items-start gap-4">
                  <Calendar className="w-8 h-8 text-blue-600 mt-1" />
                  <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">Duración del Proyecto</h3>
                    <p className="text-gray-600 font-semibold">7 semanas</p>
                    <p className="text-gray-500 text-sm mt-1">Del 06 de octubre al 23 de noviembre de 2025</p>
                  </div>
                </div>
              </div>

              <div className="bg-blue-50 rounded-xl shadow-lg p-6">
                <div className="flex items-start gap-4">
                  <CheckCircle className="w-8 h-8 text-green-600 mt-1" />
                  <div>
                    <h3 className="text-xl font-semibold text-gray-800 mb-2">Entregables Principales</h3>
                    <ul className="text-gray-600 space-y-1 ml-4 list-disc">
                      <li>Sistema IoT funcional con Raspberry Pi</li>
                      <li>Modelo de Deep Learning entrenado</li>
                      <li>Interfaz web responsiva</li>
                      <li>Documentación técnica completa</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>

            <div className="rounded-2xl overflow-hidden shadow-2xl">
              <div className="bg-gradient-to-br from-blue-500 to-purple-600 h-full flex items-center justify-center p-8">
                <div className="text-center text-white">
                  <Layers className="w-32 h-32 mx-auto mb-6" />
                  <p className="text-3xl font-bold mb-6">Arquitectura IoT</p>
                  <div className="space-y-4 text-left bg-white/10 backdrop-blur-sm rounded-xl p-6">
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 bg-green-400 rounded-full"></div>
                      <span className="text-lg">Capa de Hardware</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 bg-blue-400 rounded-full"></div>
                      <span className="text-lg">Capa de Servidor</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="w-4 h-4 bg-purple-400 rounded-full"></div>
                      <span className="text-lg">Capa de Cliente</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default Inicio