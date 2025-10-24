import { Camera, Database, Sparkles, ScanSearch, Eye } from 'lucide-react'

const Flow = () => {
  const steps = [
    { icon: <Camera className="w-8 h-8" />, title: 'Captura', color: 'from-blue-500 to-cyan-500' },
    { icon: <Database className="w-8 h-8" />, title: 'Entrenamiento', color: 'from-green-500 to-emerald-500' },
    { icon: <Sparkles className="w-8 h-8" />, title: 'Extracción', color: 'from-purple-500 to-pink-500' },
    { icon: <ScanSearch className="w-8 h-8" />, title: 'Detección', color: 'from-orange-500 to-red-500' },
    { icon: <Eye className="w-8 h-8" />, title: 'Visualización', color: 'from-pink-500 to-purple-500' }
  ]

  return (
    <div>
      <h3 className="text-3xl font-bold text-gray-800 mb-6 text-center">
        🔄 Flujo de Detección
      </h3>
      <div className="flex flex-wrap justify-center items-center gap-4">
        {steps.map((step, index) => (
          <div key={index} className="flex items-center gap-4">
            <div className={`bg-gradient-to-br ${step.color} text-white p-4 rounded-full shadow-lg`}>
              {step.icon}
            </div>
            <p className="font-semibold text-gray-700">{step.title}</p>
            {index < steps.length - 1 && <span className="text-2xl text-gray-400">→</span>}
          </div>
        ))}
      </div>
    </div>
  )
}

export default Flow