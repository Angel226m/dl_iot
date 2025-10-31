import { Shield, Cpu, Cloud, AlertTriangle, DollarSign, Clock, Users, TrendingDown, FileX, Zap, Wifi, Brain, MousePointer, ShieldCheck, Target, Calendar, CheckCircle, Layers, ArrowRight, Sparkles } from 'lucide-react'

const Inicio = () => {
  const team = [
    'Garay Torres Angel',
    'Serrrano Arias Luis', //'Figueroa Quiroz Dario',
    'Huari Mora Shandel',//'Borjas Espinoza Pierre',
    'Cama Sanchez Kevin'//'Conca Flores Omar'
  ]

  const problems = [
    { icon: <Clock className="w-7 h-7" />, title: 'Inspecciones Manuales Ineficientes', description: 'Procesos lentos que consumen muchos recursos' },
    { icon: <AlertTriangle className="w-7 h-7" />, title: 'Dificultades de Acceso', description: 'Zonas de difícil alcance requieren equipamiento especializado' },
    { icon: <Users className="w-7 h-7" />, title: 'Riesgos Humanos Elevados', description: 'Peligro para inspectores en estructuras altas' },
    { icon: <TrendingDown className="w-7 h-7" />, title: 'Detección Tardía', description: 'Las grietas se detectan cuando ya son peligrosas' },
    { icon: <DollarSign className="w-7 h-7" />, title: 'Costos Altos', description: 'Equipos especializados aumentan costos' },
    { icon: <FileX className="w-7 h-7" />, title: 'Falta de Trazabilidad', description: 'No hay registro digital del historial' },
  ]

  const solutions = [
    { icon: <Zap className="w-8 h-8" />, title: 'Automatización del Monitoreo', description: 'Sistema autónomo que reduce intervención humana', color: 'from-cyan-500 to-blue-600' },
    { icon: <Wifi className="w-8 h-8" />, title: 'Acceso Remoto IoT', description: 'Monitoreo desde cualquier ubicación en tiempo real', color: 'from-blue-500 to-indigo-600' },
    { icon: <Brain className="w-8 h-8" />, title: 'Análisis con IA', description: 'Deep Learning para detección precisa', color: 'from-indigo-500 to-blue-700' },
    { icon: <MousePointer className="w-8 h-8" />, title: 'Activación bajo Demanda', description: 'El usuario decide cuándo inspeccionar', color: 'from-blue-600 to-cyan-600' },
    { icon: <ShieldCheck className="w-8 h-8" />, title: 'Mayor Seguridad', description: 'Detección temprana previene accidentes', color: 'from-cyan-600 to-blue-500' },
  ]

  return (
    <div className="pt-16 bg-slate-950">
      {/* Hero Section - Ultra Modern */}
      <section className="relative overflow-hidden bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 py-24">
        {/* Animated Background Grid */}
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_80%_50%_at_50%_0%,#000_70%,transparent_110%)] opacity-20"></div>
        
        {/* Glow Effects */}
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl animate-pulse" style={{animationDelay: '1s'}}></div>

        <div className="relative max-w-7xl mx-auto px-6">
          <div className="text-center max-w-5xl mx-auto">
            {/* Icon Stack */}
            <div className="flex justify-center gap-4 mb-12">
              <div className="group relative">
                <div className="absolute inset-0 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-2xl blur-xl opacity-75 group-hover:opacity-100 transition duration-300"></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 p-5 rounded-2xl border border-slate-700 transform group-hover:scale-110 transition duration-300">
                  <Shield className="w-10 h-10 text-cyan-400" />
                </div>
              </div>
              <div className="group relative">
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 to-indigo-600 rounded-2xl blur-xl opacity-75 group-hover:opacity-100 transition duration-300"></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 p-5 rounded-2xl border border-slate-700 transform group-hover:scale-110 transition duration-300" style={{animationDelay: '0.1s'}}>
                  <Cpu className="w-10 h-10 text-blue-400" />
                </div>
              </div>
              <div className="group relative">
                <div className="absolute inset-0 bg-gradient-to-r from-indigo-500 to-blue-600 rounded-2xl blur-xl opacity-75 group-hover:opacity-100 transition duration-300"></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 p-5 rounded-2xl border border-slate-700 transform group-hover:scale-110 transition duration-300" style={{animationDelay: '0.2s'}}>
                  <Cloud className="w-10 h-10 text-indigo-400" />
                </div>
              </div>
            </div>

            <div className="inline-flex items-center gap-2 bg-gradient-to-r from-cyan-500/10 to-blue-500/10 border border-cyan-500/30 rounded-full px-5 py-2 mb-8">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <span className="text-cyan-400 text-sm font-semibold tracking-wide">TECNOLOGÍA DE VANGUARDIA</span>
            </div>

            <h1 className="text-6xl md:text-7xl lg:text-8xl font-black text-white mb-6 leading-tight">
              Sistema Inteligente de
              <span className="block bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-400 bg-clip-text text-transparent">
                Detección de Grietas
              </span>
            </h1>
            
            <p className="text-2xl md:text-3xl text-slate-300 mb-4 font-light">
              Potenciado por <span className="text-cyan-400 font-bold">IoT</span> y{' '}
              <span className="text-blue-400 font-bold">Deep Learning</span>
            </p>

            <div className="inline-flex items-center gap-3 bg-gradient-to-r from-cyan-600 to-blue-600 text-white px-10 py-5 rounded-2xl shadow-2xl shadow-blue-500/50 mb-16 hover:shadow-blue-500/70 transition-all duration-300 transform hover:scale-105">
              <Shield className="w-7 h-7" />
              <p className="text-xl font-bold">CrackGuard: Detección Inteligente, Acceso sin Riesgos</p>
            </div>

            {/* Team Grid */}
            <div className="mt-20">
              <div className="inline-flex items-center gap-2 mb-10">
                <div className="h-px w-12 bg-gradient-to-r from-transparent to-cyan-500"></div>
                <h3 className="text-2xl font-bold text-white tracking-tight">Equipo de Desarrollo</h3>
                <div className="h-px w-12 bg-gradient-to-l from-transparent to-cyan-500"></div>
              </div>
              
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                {team.map((member, index) => (
                  <div key={index} className="group relative">
                    <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/20 to-blue-600/20 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
                    <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-8 hover:border-cyan-500/50 transition-all duration-300 transform group-hover:-translate-y-2">
                      <div className="w-20 h-20 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-2xl mx-auto mb-5 flex items-center justify-center shadow-lg shadow-cyan-500/50 relative overflow-hidden">
                        <div className="absolute inset-0 bg-white/20 transform -skew-x-12 group-hover:translate-x-full transition-transform duration-700"></div>
                        <span className="text-3xl font-black text-white relative z-10">{member.split(' ')[0][0]}</span>
                      </div>
                      <p className="text-base font-semibold text-white mb-1">{member.split(' ')[0]} {member.split(' ')[1]}</p>
                      <p className="text-sm text-slate-400">{member.split(' ').slice(2).join(' ')}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Problems Section */}
      <section className="relative bg-slate-900 py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-red-500/10 border border-red-500/30 rounded-full px-5 py-2 mb-6">
              <AlertTriangle className="w-4 h-4 text-red-400" />
              <span className="text-red-400 text-sm font-semibold tracking-wide">DESAFÍOS ACTUALES</span>
            </div>
            <h2 className="text-5xl md:text-6xl font-black text-white mb-6">Problemática</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Los métodos tradicionales presentan múltiples desafíos que comprometen la seguridad estructural
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {problems.map((problem, index) => (
              <div key={index} className="group relative">
                <div className="absolute inset-0 bg-gradient-to-br from-red-500/10 to-orange-600/10 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-8 hover:border-red-500/50 transition-all duration-300 h-full">
                  <div className="bg-gradient-to-br from-red-500 to-orange-600 text-white p-4 rounded-xl mb-5 inline-flex shadow-lg shadow-red-500/30">
                    {problem.icon}
                  </div>
                  <h3 className="text-xl font-bold text-white mb-3">{problem.title}</h3>
                  <p className="text-slate-400 leading-relaxed">{problem.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Solution Section */}
      <section className="relative bg-slate-950 py-24">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>
        
        <div className="relative max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-green-500/10 border border-green-500/30 rounded-full px-5 py-2 mb-6">
              <CheckCircle className="w-4 h-4 text-green-400" />
              <span className="text-green-400 text-sm font-semibold tracking-wide">NUESTRA SOLUCIÓN</span>
            </div>
            <h2 className="text-5xl md:text-6xl font-black text-white mb-6">Tecnología CrackGuard</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Integración perfecta de IoT y Deep Learning para revolucionar la inspección estructural
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {solutions.map((solution, index) => (
              <div key={index} className="group relative">
                <div className={`absolute inset-0 bg-gradient-to-br ${solution.color} rounded-2xl blur-xl opacity-0 group-hover:opacity-50 transition duration-500`}></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-8 hover:border-cyan-500/50 transition-all duration-300 h-full transform group-hover:-translate-y-2">
                  <div className={`bg-gradient-to-br ${solution.color} text-white p-4 rounded-xl mb-5 inline-flex shadow-lg`}>
                    {solution.icon}
                  </div>
                  <h3 className="text-xl font-bold text-white mb-3">{solution.title}</h3>
                  <p className="text-slate-400 leading-relaxed">{solution.description}</p>
                  <ArrowRight className="w-5 h-5 text-cyan-400 mt-4 transform group-hover:translate-x-2 transition-transform" />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Scope Section */}
      <section className="relative bg-slate-900 py-24">
        <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/30 rounded-full px-5 py-2 mb-6">
              <Target className="w-4 h-4 text-blue-400" />
              <span className="text-blue-400 text-sm font-semibold tracking-wide">ALCANCE</span>
            </div>
            <h2 className="text-5xl md:text-6xl font-black text-white mb-6">Plan de Ejecución</h2>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="space-y-6">
              {[
                { icon: Target, title: 'Desarrollo del MVP', desc: 'Enfoque en detección de grietas con un Producto Mínimo Viable', color: 'cyan' },
                { icon: Layers, title: 'Flujo Completo', desc: 'Captura → Análisis con IA → Visualización de resultados', color: 'blue' },
                { icon: Calendar, title: 'Duración: 7 semanas', desc: 'Del 06 de octubre al 23 de noviembre de 2025', color: 'indigo' },
                { icon: CheckCircle, title: 'Entregables', desc: 'Sistema IoT • Modelo DL • Interfaz Web • Documentación', color: 'green' },
              ].map((item, index) => (
                <div key={index} className="group relative">
                  <div className={`absolute inset-0 bg-${item.color}-500/10 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition duration-500`}></div>
                  <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500/50 transition-all duration-300">
                    <div className="flex items-start gap-4">
                      <div className={`bg-gradient-to-br from-${item.color}-500 to-${item.color}-600 p-3 rounded-xl`}>
                        <item.icon className="w-6 h-6 text-white" />
                      </div>
                      <div>
                        <h3 className="text-xl font-bold text-white mb-2">{item.title}</h3>
                        <p className="text-slate-400">{item.desc}</p>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/20 to-blue-600/20 rounded-3xl blur-2xl opacity-75 group-hover:opacity-100 transition duration-500"></div>
              <div className="relative bg-gradient-to-br from-slate-800 via-slate-900 to-slate-950 border border-slate-700 rounded-3xl p-12 h-full flex flex-col justify-center">
                <Layers className="w-24 h-24 text-cyan-400 mx-auto mb-8" />
                <p className="text-4xl font-black text-white text-center mb-8">Arquitectura IoT</p>
                <div className="space-y-5">
                  {['Capa de Hardware', 'Capa de Servidor', 'Capa de Cliente'].map((layer, i) => (
                    <div key={i} className="flex items-center gap-4 bg-slate-800/50 rounded-xl p-4 border border-slate-700 hover:border-cyan-500/50 transition-all duration-300">
                      <div className={`w-3 h-3 rounded-full ${i === 0 ? 'bg-cyan-400' : i === 1 ? 'bg-blue-400' : 'bg-indigo-400'}`}></div>
                      <span className="text-lg font-semibold text-white">{layer}</span>
                    </div>
                  ))}
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