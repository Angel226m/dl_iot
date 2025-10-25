import { Cpu, Server, Monitor, Users, Building2, Shield, Code, Server as ServerIcon, Package, GraduationCap, Camera, Database, Sparkles, ScanSearch, Eye, Calendar, CheckCircle, Clock, ArrowRight, Layers } from 'lucide-react'

const Funcionamiento = () => {
  const layers = [
    {
      icon: <Cpu className="w-10 h-10" />,
      title: 'Capa de Hardware',
      technology: 'Raspberry Pi 4 + Arducam',
      color: 'from-cyan-500 to-blue-600',
      description: 'Dispositivos IoT para captura de imágenes en tiempo real',
      features: ['Raspberry Pi 4', 'Cámara de alta resolución', 'Sensores ambientales'],
    },
    {
      icon: <Server className="w-10 h-10" />,
      title: 'Capa de Servidor',
      technology: 'Python + Flask + TensorFlow',
      color: 'from-blue-500 to-indigo-600',
      description: 'Procesamiento y análisis con modelos de Deep Learning',
      features: ['API Flask', 'Modelo U-Net', 'Optimización con TensorFlow'],
    },
    {
      icon: <Monitor className="w-10 h-10" />,
      title: 'Capa de Cliente',
      technology: 'React + TypeScript + Tailwind',
      color: 'from-indigo-500 to-cyan-600',
      description: 'Interfaz web responsiva para visualización de resultados',
      features: ['React 18', 'TypeScript', 'Diseño adaptativo'],
    },
  ]

  const stakeholders = [
    { icon: <Users className="w-8 h-8" />, title: 'Ingenieros Estructurales', description: 'Análisis y mantenimiento de estructuras', color: 'from-cyan-500 to-blue-500' },
    { icon: <Building2 className="w-8 h-8" />, title: 'Propietarios', description: 'Gestión y monitoreo de edificaciones', color: 'from-green-500 to-emerald-500' },
    { icon: <Shield className="w-8 h-8" />, title: 'Autoridades Municipales', description: 'Regulación y cumplimiento de normas', color: 'from-red-500 to-orange-500' },
    { icon: <Code className="w-8 h-8" />, title: 'Equipo de Desarrollo', description: 'Diseño y desarrollo de CrackGuard', color: 'from-blue-500 to-indigo-500' },
    { icon: <ServerIcon className="w-8 h-8" />, title: 'Administradores de Servidores', description: 'Mantenimiento de infraestructura IT', color: 'from-yellow-500 to-orange-500' },
    { icon: <Package className="w-8 h-8" />, title: 'Proveedores', description: 'Suministro de hardware y software', color: 'from-cyan-500 to-teal-500' },
    { icon: <GraduationCap className="w-8 h-8" />, title: 'Universidad Nacional de Cañete', description: 'Supervisión académica y evaluación', color: 'from-indigo-500 to-blue-600' },
  ]

  const flowSteps = [
    { icon: <Camera className="w-10 h-10" />, title: 'Captura de Datos', description: 'Raspberry Pi con Arducam captura imágenes de estructuras', number: 1, color: 'from-cyan-500 to-blue-500' },
    { icon: <Database className="w-10 h-10" />, title: 'Entrenamiento del Modelo', description: 'Modelo U-Net entrenado con dataset de grietas', number: 2, color: 'from-green-500 to-emerald-500' },
    { icon: <Sparkles className="w-10 h-10" />, title: 'Extracción de Características', description: 'Red neuronal identifica patrones relevantes', number: 3, color: 'from-blue-500 to-indigo-500' },
    { icon: <ScanSearch className="w-10 h-10" />, title: 'Detección y Segmentación', description: 'Algoritmo U-Net delimita áreas con grietas', number: 4, color: 'from-orange-500 to-red-500' },
    { icon: <Eye className="w-10 h-10" />, title: 'Visualización de Resultados', description: 'Interfaz web muestra resultados con métricas', number: 5, color: 'from-indigo-500 to-cyan-500' },
  ]

  const timeline = [
    { week: 'Semana 1-2', dates: '06 Oct - 19 Oct 2025', activities: ['Investigación y planificación', 'Definición de requisitos', 'Diseño de arquitectura'], responsible: 'Todo el equipo', deliverable: 'Documento de especificaciones' },
    { week: 'Semana 3-4', dates: '20 Oct - 02 Nov 2025', activities: ['Configuración de Raspberry Pi', 'Desarrollo backend Flask', 'Recolección de dataset'], responsible: 'Garay, Figueroa', deliverable: 'Prototipo de hardware y API' },
    { week: 'Semana 5', dates: '03 Nov - 09 Nov 2025', activities: ['Entrenamiento del modelo U-Net', 'Optimización de hiperparámetros', 'Validación del modelo'], responsible: 'Borjas, Conca', deliverable: 'Modelo entrenado' },
    { week: 'Semana 6', dates: '10 Nov - 16 Nov 2025', activities: ['Desarrollo del frontend React', 'Integración con backend', 'Diseño de interfaz'], responsible: 'Figueroa, Garay', deliverable: 'Aplicación web funcional' },
    { week: 'Semana 7', dates: '17 Nov - 23 Nov 2025', activities: ['Pruebas integrales', 'Corrección de bugs', 'Documentación final', 'Preparación de presentación'], responsible: 'Todo el equipo', deliverable: 'Sistema completo y documentación' },
  ]

  return (
    <div className="pt-16 bg-slate-950">
      {/* Arquitectura de Tres Capas */}
      <section className="relative bg-slate-900 py-20">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-cyan-500/10 border border-cyan-500/30 rounded-full px-5 py-2 mb-6">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span className="text-cyan-400 text-sm font-semibold tracking-wide">ARQUITECTURA</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">Arquitectura de Tres Capas</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Sistema modular con integración escalable y eficiente
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            {layers.map((layer, index) => (
              <div key={index} className="group relative">
                <div className={`absolute inset-0 bg-gradient-to-br ${layer.color} rounded-2xl blur-xl opacity-0 group-hover:opacity-50 transition duration-500`}></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500/50 transition-all duration-300 h-full">
                  <div className={`bg-gradient-to-br ${layer.color} text-white p-4 rounded-xl mb-4 inline-flex shadow-lg`}>
                    {layer.icon}
                  </div>
                  <h3 className="text-xl font-bold text-white mb-2">{layer.title}</h3>
                  <p className="text-sm font-semibold text-cyan-400 mb-3">{layer.technology}</p>
                  <p className="text-slate-400 mb-4 leading-relaxed">{layer.description}</p>
                  <div className="space-y-2">
                    {layer.features.map((feature, idx) => (
                      <div key={idx} className="flex items-center gap-2 text-sm text-slate-300">
                        <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full"></div>
                        <span>{feature}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Diagrama de flujo */}
          <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-8 hover:border-cyan-500/50 transition-all duration-300">
            <h3 className="text-2xl font-bold text-center mb-8 text-white">Flujo de Comunicación</h3>
            <div className="flex flex-col md:flex-row items-center justify-center gap-6 md:gap-8">
              <div className="flex flex-col items-center group">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full blur-xl opacity-75 group-hover:opacity-100 transition duration-300"></div>
                  <div className="relative w-20 h-20 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center text-white font-bold text-2xl shadow-lg shadow-cyan-500/50">
                    1
                  </div>
                </div>
                <p className="mt-3 font-semibold text-white">Hardware</p>
              </div>
              <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
              <div className="text-4xl text-slate-600 md:hidden">↓</div>
              <div className="flex flex-col items-center group">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full blur-xl opacity-75 group-hover:opacity-100 transition duration-300"></div>
                  <div className="relative w-20 h-20 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-full flex items-center justify-center text-white font-bold text-2xl shadow-lg shadow-blue-500/50">
                    2
                  </div>
                </div>
                <p className="mt-3 font-semibold text-white">Servidor</p>
              </div>
              <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
              <div className="text-4xl text-slate-600 md:hidden">↓</div>
              <div className="flex flex-col items-center group">
                <div className="relative">
                  <div className="absolute inset-0 bg-gradient-to-br from-indigo-500 to-cyan-600 rounded-full blur-xl opacity-75 group-hover:opacity-100 transition duration-300"></div>
                  <div className="relative w-20 h-20 bg-gradient-to-br from-indigo-500 to-cyan-600 rounded-full flex items-center justify-center text-white font-bold text-2xl shadow-lg shadow-indigo-500/50">
                    3
                  </div>
                </div>
                <p className="mt-3 font-semibold text-white">Cliente</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stakeholders */}
      <section className="relative bg-slate-950 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/30 rounded-full px-5 py-2 mb-6">
              <Users className="w-4 h-4 text-blue-400" />
              <span className="text-blue-400 text-sm font-semibold tracking-wide">STAKEHOLDERS</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">Actores y Stakeholders</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Partes interesadas clave en el desarrollo de CrackGuard
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
            {stakeholders.map((stakeholder, index) => (
              <div key={index} className="group relative">
                <div className={`absolute inset-0 bg-gradient-to-br ${stakeholder.color} rounded-2xl blur-xl opacity-0 group-hover:opacity-50 transition duration-500`}></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500/50 transition-all duration-300 h-full transform group-hover:-translate-y-2">
                  <div className={`bg-gradient-to-br ${stakeholder.color} text-white p-3 rounded-xl mb-4 inline-flex shadow-lg`}>
                    {stakeholder.icon}
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">{stakeholder.title}</h3>
                  <p className="text-slate-400 text-sm leading-relaxed">{stakeholder.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Flujo de Detección */}
      <section className="relative bg-slate-900 py-20">
        <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:4rem_4rem] opacity-20"></div>
        
        <div className="relative max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-green-500/10 border border-green-500/30 rounded-full px-5 py-2 mb-6">
              <Sparkles className="w-4 h-4 text-green-400" />
              <span className="text-green-400 text-sm font-semibold tracking-wide">PROCESO</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">Flujo de Detección</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Proceso completo desde la captura hasta la visualización de resultados
            </p>
          </div>

          <div className="space-y-6">
            {flowSteps.map((step, index) => (
              <div key={index} className="group">
                <div className="flex flex-col md:flex-row items-start md:items-center gap-6">
                  <div className="relative flex-shrink-0">
                    <div className={`absolute inset-0 bg-gradient-to-br ${step.color} rounded-full blur-xl opacity-75 group-hover:opacity-100 transition duration-300`}></div>
                    <div className={`relative w-20 h-20 bg-gradient-to-br ${step.color} rounded-full flex items-center justify-center text-white shadow-lg`}>
                      {step.icon}
                    </div>
                  </div>
                  
                  <div className="flex-grow">
                    <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500/50 transition-all duration-300">
                      <div className="flex items-center gap-3 mb-2">
                        <span className={`text-3xl font-black bg-gradient-to-br ${step.color} bg-clip-text text-transparent`}>
                          {step.number}
                        </span>
                        <h3 className="text-xl md:text-2xl font-bold text-white">{step.title}</h3>
                      </div>
                      <p className="text-slate-400 text-base md:text-lg leading-relaxed">{step.description}</p>
                    </div>
                  </div>
                </div>

                {index < flowSteps.length - 1 && (
                  <div className="flex justify-center my-4">
                    <div className="text-4xl text-slate-700">↓</div>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="mt-16 relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-blue-600/20 rounded-3xl blur-2xl opacity-75 group-hover:opacity-100 transition duration-500"></div>
            <div className="relative bg-gradient-to-br from-slate-800 via-slate-900 to-slate-950 border border-slate-700 rounded-3xl p-8 md:p-12">
              <h3 className="text-3xl font-black text-white text-center mb-8">Deep Learning en Acción</h3>
              <div className="flex flex-col md:flex-row justify-around items-center gap-8 md:gap-6">
                <div className="text-center group/item">
                  <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl mb-3 hover:border-cyan-500/50 transition-all duration-300">
                    <Database className="w-16 h-16 text-cyan-400 mx-auto" />
                  </div>
                  <p className="font-bold text-white text-lg">Entrada</p>
                  <p className="text-sm text-slate-400">Imágenes</p>
                </div>
                <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
                <div className="text-4xl text-slate-600 md:hidden">↓</div>
                <div className="text-center group/item">
                  <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl mb-3 hover:border-blue-500/50 transition-all duration-300">
                    <Cpu className="w-16 h-16 text-blue-400 mx-auto" />
                  </div>
                  <p className="font-bold text-white text-lg">Procesamiento</p>
                  <p className="text-sm text-slate-400">U-Net Model</p>
                </div>
                <ArrowRight className="w-8 h-8 text-slate-600 hidden md:block" />
                <div className="text-4xl text-slate-600 md:hidden">↓</div>
                <div className="text-center group/item">
                  <div className="bg-slate-800 border border-slate-700 p-6 rounded-2xl mb-3 hover:border-indigo-500/50 transition-all duration-300">
                    <Eye className="w-16 h-16 text-indigo-400 mx-auto" />
                  </div>
                  <p className="font-bold text-white text-lg">Salida</p>
                  <p className="text-sm text-slate-400">Detección</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Cronograma */}
      <section className="relative bg-slate-950 py-20">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="text-center mb-16">
            <div className="inline-flex items-center gap-2 bg-blue-500/10 border border-blue-500/30 rounded-full px-5 py-2 mb-6">
              <Calendar className="w-4 h-4 text-blue-400" />
              <span className="text-blue-400 text-sm font-semibold tracking-wide">PLANIFICACIÓN</span>
            </div>
            <h2 className="text-4xl md:text-5xl lg:text-6xl font-black text-white mb-6">Cronograma del Proyecto</h2>
            <p className="text-xl text-slate-400 max-w-3xl mx-auto">
              Planificación de 7 semanas (06 Oct - 23 Nov 2025)
            </p>
          </div>

          <div className="space-y-6">
            {timeline.map((item, index) => (
              <div key={index} className="group relative">
                <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-blue-600/10 rounded-2xl blur-xl opacity-0 group-hover:opacity-100 transition duration-500"></div>
                <div className="relative bg-gradient-to-br from-slate-800 to-slate-900 border border-slate-700 rounded-2xl p-6 hover:border-cyan-500/50 transition-all duration-300">
                  <div className="flex flex-col lg:flex-row gap-6">
                    <div className="flex-shrink-0">
                      <div className="relative">
                        <div className="absolute inset-0 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full blur-xl opacity-75"></div>
                        <div className="relative w-20 h-20 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-full flex items-center justify-center text-white font-black text-2xl shadow-lg shadow-cyan-500/50">
                          {index + 1}
                        </div>
                      </div>
                    </div>

                    <div className="flex-grow space-y-4">
                      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                        <h3 className="text-2xl font-bold text-white">{item.week}</h3>
                        <div className="flex items-center gap-2 text-slate-400">
                          <Calendar className="w-5 h-5 text-cyan-400" />
                          <span className="font-medium">{item.dates}</span>
                        </div>
                      </div>

                      <div className="grid sm:grid-cols-2 gap-4">
                        <div className="bg-slate-900/50 border border-slate-700 rounded-xl p-4">
                          <h4 className="font-semibold text-white mb-3 flex items-center gap-2">
                            <Clock className="w-4 h-4 text-blue-400" />
                            Actividades:
                          </h4>
                          <ul className="space-y-2">
                            {item.activities.map((activity, idx) => (
                              <li key={idx} className="flex items-start gap-2 text-slate-300 text-sm">
                                <CheckCircle className="w-4 h-4 text-green-400 flex-shrink-0 mt-0.5" />
                                <span>{activity}</span>
                              </li>
                            ))}
                          </ul>
                        </div>

                        <div className="space-y-4">
                          <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-xl p-4">
                            <h4 className="font-semibold text-white mb-2 text-sm">👥 Responsables:</h4>
                            <p className="text-cyan-400 font-medium">{item.responsible}</p>
                          </div>
                          
                          <div className="bg-blue-500/10 border border-blue-500/30 rounded-xl p-4">
                            <h4 className="font-semibold text-white mb-2 text-sm">📦 Entregable:</h4>
                            <p className="text-blue-400 font-medium">{item.deliverable}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  )
}

export default Funcionamiento